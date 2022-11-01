import os
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax import jit, vmap
import optax
import time
import pandas as pd
import numpy as np
import time
import math
from pathlib import Path
import utils
import sklearn.metrics
import psutil
import resource

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def generate_gaussian(key, shape, scale=0.1):
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)


def generate_sparsity_mask(key, layer_sizes, type, sparsity):
    sparsity_mask = []
    if type == "activity":
        for m in layer_sizes:
            key, _ = jax.random.split(key)
            sparsity_mask.append(
                jax.random.categorical(
                    key,
                    shape=(m, 1),
                    logits=jnp.log(jnp.array([1 - sparsity, sparsity])),
                )
            )

    elif type == "weight":
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, _ = jax.random.split(key)
            sparsity_mask.append(
                jax.random.categorical(
                    key,
                    shape=(n, m),
                    logits=jnp.log(jnp.array([1 - sparsity, sparsity])),
                )
            )

    return sparsity_mask


def generate_plasticity_mask(upto_ith_order):
    plasticity_mask = [0 for _ in range(27)]
    for index in range(27):
        pxyw = utils.A_index_to_powers(index)
        if sum(pxyw) <= upto_ith_order and sum(pxyw) > 1:
            plasticity_mask[index] = 1

    return jnp.array(plasticity_mask)


def generate_A_teacher(plasticity_rule):
    A_teacher = np.zeros((27,))
    if plasticity_rule == "oja":
        A_teacher[utils.powers_to_A_index(1, 1, 0)] = 1
        A_teacher[utils.powers_to_A_index(0, 2, 1)] = -1

    elif plasticity_rule == "hebbian":
        A_teacher[utils.powers_to_A_index(1, 1, 0)] = 1

    elif plasticity_rule == "random":
        A_teacher[utils.powers_to_A_index(1, 1, 0)] = np.random.rand()
        A_teacher[utils.powers_to_A_index(0, 2, 1)] = -1 * np.random.rand()

    else:
        raise Exception("plasticity rule must be either oja, hebbian or random")

    return jnp.array(A_teacher)


# IMP: don't jit unrolling computation graph. Can try w lax.scan and then jit
def generate_weight_trajec(key, noise_scale, x_data, weights, A):
    num_trajec, len_trajec = x_data.shape[0], x_data.shape[1]
    weight_trajec_data = [[] for _ in range(num_trajec)]
    w_init = [w.copy() for w in weights]

    for i in range(num_trajec):
        weights = [w.copy() for w in w_init]
        for j in range(len_trajec):
            key, _ = jax.random.split(key)
            weight_trajec_data[i].append([w.copy() for w in weights])
            weights = update_weights(weights, x_data[i][j], A)
            for layer in range(len(weights)):
                weight_trajec_data[i][-1][layer] += noise_scale * jax.random.normal(
                    key, weights[layer].shape
                )

    return weight_trajec_data


def generate_activity_trajec(key, noise_scale, x_data, weights, A):
    num_trajec, len_trajec = x_data.shape[0], x_data.shape[1]
    activity_trajec_data = [[] for _ in range(num_trajec)]
    w_init = [w.copy() for w in weights]

    for i in range(num_trajec):
        weights = [w.copy() for w in w_init]
        for j in range(len_trajec):
            key, _ = jax.random.split(key)
            act = forward(weights, x_data[i][j])
            for layer in range(len(act)):
                act[layer] += noise_scale * jax.random.normal(key, act[layer].shape)
            activity_trajec_data[i].append(act)
            weights = update_weights(weights, x_data[i][j], A)

    return activity_trajec_data


def calc_loss_weight_trajec_(
    sparsity_mask,
    plasticity_mask,
    l1_lmbda,
    weights,
    x,
    A,
    weight_trajectory,
):
    loss = 0

    for i in range(len(weight_trajectory)):
        teacher_weights = weight_trajectory[i]

        for j in range(len(weights)):
            loss_mat = optax.l2_loss(weights[j], teacher_weights[j])
            loss += jnp.mean(jnp.multiply(sparsity_mask[j], loss_mat))
        weights = update_weights(weights, x[i], A)
    loss /= len(weight_trajectory)
    # add L1 regularization term to enforce sparseness
    loss += l1_lmbda * jnp.sum(jnp.absolute(A * plasticity_mask))

    return loss


def calc_loss_activity_trajec_(
    sparsity_mask,
    plasticity_mask,
    l1_lmbda,
    weights,
    x,
    A,
    activity_trajectory,
):
    loss = 0
    use_input = True

    for i in range(len(activity_trajectory)):
        loss_t = []
        act = forward(weights, x[i])
        teacher_act = activity_trajectory[i]
        for j in range(len(act)):
            loss_mat = optax.l2_loss(act[j], teacher_act[j])
            loss_t.append(jnp.mean(jnp.multiply(sparsity_mask[j], loss_mat)))
        if not use_input:
            loss_t.pop(0)
        loss += sum(loss_t)
        weights = update_weights(weights, x[i], A)

    loss /= len(activity_trajectory)
    # add L1 regularization term to enforce sparseness
    loss += l1_lmbda * jnp.sum(jnp.absolute(A * plasticity_mask))

    return loss


def update_weights_(plasticity_mask, weights, x, A):
    # note: check if this can be simplified with jax.tree_map()
    act = forward(weights, x)
    A = A * plasticity_mask
    for layer in range(len(weights)):
        dw = 0
        for index in range(len(A)):
            i, j, k = utils.A_index_to_powers(index)
            dw += A[index] * jnp.multiply(
                jnp.outer(act[layer + 1] ** j, act[layer] ** i), weights[layer] ** k
            )

        if dw.shape != weights[layer].shape:
            raise Exception(
                "dw and w should be of the same shape to prevent broadcasting while adding"
            )
        weights[layer] += dw

    return weights


def forward_(non_linear, weights, x):
    act = [jnp.expand_dims(x, 1)]
    for layer in range(len(weights)):
        h = jnp.dot(weights[layer], act[-1])
        if non_linear:
            act.append(jax.nn.sigmoid(h))
        else:
            act.append(h)
    return act


def calculate_r2_score(key, A_student, A_teacher, layer_sizes, x_data):

    weights = []
    num_test_trajec, len_test_trajec = x_data.shape[0], x_data.shape[1]

    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
        weights.append(generate_gaussian(key, (n, m), scale=1 / (m + n)))

    true_trajec_data_w = generate_weight_trajec(
        jax.random.PRNGKey(0), 0.0, x_data, weights, A_teacher
    )

    pred_trajec_data_w = generate_weight_trajec(
        jax.random.PRNGKey(0), 0.0, x_data, weights, A_student
    )
    y = [
        jnp.concatenate(true_trajec_data_w[i][j], axis=None)
        for i in range(num_test_trajec)
        for j in range(len_test_trajec)
    ]
    y_pred = [
        jnp.concatenate(pred_trajec_data_w[i][j], axis=None)
        for i in range(num_test_trajec)
        for j in range(len_test_trajec)
    ]
    r2_score = sklearn.metrics.r2_score(y, y_pred)
    return r2_score


def main():
    (
        input_dim,
        output_dim,
        hidden_layers,
        hidden_neurons,
        non_linear,  # True/False
        plasticity_rule,  # Hebb, Oja, Random
        meta_epochs,
        num_trajec,
        len_trajec,
        type,  # activity trace, weight trace
        upto_ith_order,
        l1_lmbda,
        sparsity,
        noise_scale,
        log_expdata,
        output_file,
        jobid,
    ) = utils.parse_args()
    key = jax.random.PRNGKey(jobid)

    device = jax.lib.xla_bridge.get_backend().platform  # are we running on CPU or GPU?

    path = "explogs/"
    layer_sizes = [input_dim]

    for _ in range(hidden_layers):
        layer_sizes.append(hidden_neurons)
        if hidden_neurons == -1:
            raise Exception("specify the number of hidden neurons in the network")

    layer_sizes.append(output_dim)
    print("network architecture: ", layer_sizes)
    print("platform: ", device)
    teacher_weights, student_weights = [], []

    # same random initialization of the weights at the start for student and teacher network
    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
        teacher_weights.append(generate_gaussian(key, (n, m), scale=1 / (m + n)))
        student_weights.append(generate_gaussian(key, (n, m), scale=1 / (m + n)))

    A_teacher = generate_A_teacher(plasticity_rule)
    key, key2 = jax.random.split(key)
    A_student = generate_gaussian(key2, (27,), scale=1e-3)
    # A_student = jnp.zeros((27,))
    # A_student = A_student.at[4].set(1)
    # A_student = A_student.at[15].set(-1)

    # sparsity of 0.9 retains ~90% of the trace
    sparsity_mask = generate_sparsity_mask(key, layer_sizes, type, sparsity)
    plasticity_mask = generate_plasticity_mask(upto_ith_order)
    num_meta_params = sum(plasticity_mask)

    global forward, update_weights
    forward = jit(Partial(forward_, non_linear))
    update_weights = jit(Partial((update_weights_), plasticity_mask))

    if type == "weight":
        calc_loss_trajec = Partial(
            (calc_loss_weight_trajec_), sparsity_mask, plasticity_mask, l1_lmbda
        )
        generate_trajec = generate_weight_trajec
    elif type == "activity":
        calc_loss_trajec = Partial(
            (calc_loss_activity_trajec_), sparsity_mask, plasticity_mask, l1_lmbda
        )
        generate_trajec = generate_activity_trajec
    else:
        raise Exception("only weight trace or activity trace allowed")

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(A_student)

    expdata = {
        "A_" + str(i) + str(j) + str(k): []
        for i in range(3)
        for j in range(3)
        for k in range(3)
    }
    expdata.update(
        {
            "loss": [],
            "mean_grad_norm": [],
            "backprop_time": [],
            "r2_score": [],
            "epoch": jnp.arange(meta_epochs) + 1,
        }
    )

    x_data = generate_gaussian(key, (num_trajec, len_trajec, input_dim), scale=0.1)
    # generate same trajectory on repeat
    # x_data = jnp.repeat(x_data, num_trajec, axis=0).reshape((num_trajec, len_trajec, input_dim), order='F')

    key, _ = jax.random.split(key)
    start_time = time.time()
    print("generating trajec w nested for loop")
    teacher_trajec_data = generate_activity_trajec(
        key, noise_scale, x_data, teacher_weights, A_teacher
    )

    process = psutil.Process(os.getpid())
    print(
        "generated training data in %.2f s \nstarting training..."
        % (time.time() - start_time)
    )

    for _ in range(meta_epochs):
        start_time = time.time()
        expdata["loss"].append(0)
        expdata["mean_grad_norm"].append(0)
        for idx in range(len(A_student)):
            pi, pj, pk = utils.A_index_to_powers(idx)
            expdata["A_{}{}{}".format(pi, pj, pk)].append(A_student[idx])

        for j in range(num_trajec):
            x = x_data[j]
            teacher_trajec = teacher_trajec_data[j]

            loss_T, grads = jax.value_and_grad(calc_loss_trajec, argnums=2)(
                student_weights, x, A_student, teacher_trajec
            )
            # loss_T = calc_loss_trajec(student_weights, x, A_student, teacher_trajec)

            expdata["mean_grad_norm"][-1] += jnp.linalg.norm(grads)
            expdata["loss"][-1] += loss_T

            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                A_student,
            )
            A_student = optax.apply_updates(A_student, updates)

        expdata["loss"][-1] = round(math.sqrt(expdata["loss"][-1] / num_trajec), 6)
        expdata["mean_grad_norm"][-1] = round(
            math.sqrt(expdata["mean_grad_norm"][-1] / num_trajec), 6
        )
        expdata["backprop_time"].append(
            round((time.time() - start_time) / num_trajec, 3)
        )
        key, key2 = jax.random.split(key)
        test_data = generate_gaussian(key, (25, len_trajec, input_dim), scale=0.1)
        expdata["r2_score"].append(
            calculate_r2_score(key2, A_student, A_teacher, layer_sizes, test_data)
        )
        print("r2_score: ", expdata["r2_score"][-1])

        print(
            "sqrt avg.loss (across num_trajectories, len_trajectory)",
            expdata["loss"][-1],
        )
        print()

    print("Mem usage: ", round(process.memory_info().rss / 10**6), "MB")
    df = pd.DataFrame(expdata)
    pd.set_option("display.max_columns", None)

    (
        df["input_dim"],
        df["output_dim"],
        df["hidden_layers"],
        df["hidden_neurons"],
        # df["non_linear"],
        # df["plasticity_rule"],
        df["meta_epochs"],
        df["num_trajec"],
        df["len_trajec"],
        df["type"],
        df["upto_ith_order"],
        df["num_meta_params"],
        df["l1_lmbda"],
        df["sparsity"],
        df["noise_scale"],
        df["device"],
        df["jobid"],
    ) = (
        input_dim,
        output_dim,
        hidden_layers,
        hidden_neurons,
        # non_linear,
        # plasticity_rule,
        meta_epochs,
        num_trajec,
        len_trajec,
        type,
        upto_ith_order,
        num_meta_params,
        l1_lmbda,
        sparsity,
        noise_scale,
        device,
        jobid,
    )

    print(df.head(5))

    if log_expdata:
        use_header = False
        Path(path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(path + "{}.csv".format(output_file)):
            use_header = True

        df.to_csv(path + "{}.csv".format(output_file), mode="a", header=use_header)
        print("wrote training logs to disk")


if __name__ == "__main__":
    main()

###################################################################
#################         Code Blocks          ####################
###################################################################

# def vec_generate_activity_trajec(key, noise_scale, x, weights, A):
#     len_trajec = x.shape[0]
#     activity_trajec_data = []

#     for j in range(len_trajec):
#         key, _ = jax.random.split(key)
#         act = forward(weights, x[j])
#         for layer in range(len(act)):
#             act[layer] += noise_scale * jax.random.normal(key, act[layer].shape)
#         activity_trajec_data.append(act)
#         weights = update_weights(weights, x[j], A)

#     return activity_trajec_data

# teacher_trajec_data = vmap(vec_generate_activity_trajec, in_axes=(None, None, 0, None, None), out_axes=0)(
#     key, noise_scale, x_data, teacher_weights, A_teacher
# )

# print("max usage", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
