from re import L
import jax
import jax.numpy as jnp
import optax
import time
import pandas as pd

def generate_gaussian(key, shape, scale=0.1):
    return scale * jax.random.normal(key, (shape))

@jax.jit
def generate_weight_trajectory(x, weights, A):
    weight_trajectory = []

    for i in range(len(x)):
        weights = update_weights(x[i], weights, A)
        weight_trajectory.append(weights)

    return weight_trajectory

@jax.jit
def generate_activity_trajectory(x, weights, A):
    activity_trajectory = []

    for i in range(len(x)):
        weights = update_weights(x[i], weights, A)
        y = forward_pass(x[i], weights)
        activity_trajectory.append(y)

    return activity_trajectory

@jax.jit
def weight_trajectory_loss(x, weights, A, weight_trajectory):
    loss = 0

    for i in range(len(x)):
        weights = update_weights(x[i], weights, A)
        loss += jnp.mean(optax.l2_loss(weights, weight_trajectory[i]))
    return loss

@jax.jit
def activity_trajectory_loss(x, weights, A, activity_trajectory):
    loss = 0

    for i in range(len(x)):
        weights = update_weights(x[i], weights, A)
        y = forward_pass(x[i], weights)
        loss += jnp.mean(optax.l2_loss(y, activity_trajectory[i]))
    return loss

def update_weights(x, weights, A):
    y = forward_pass(x, weights)
    dw = A[0] * x * y + A[1] * jnp.multiply(y**2, weights)
    weights += dw
    return weights

def forward_pass(x, weights):
    return jax.nn.sigmoid(jnp.dot(x, weights))

"""
document the following:
    m/n neurons
    trajectory update time
    # learnable params
    underlying update rule
    type: activity / weight trace
    non-linearity
"""

def main():
    key = jax.random.PRNGKey(0)
    meta_epochs = 3
    num_trajectories = 10
    length = 10 
    m, n = 10, 1
    teacher_weights = generate_gaussian(key, (m, n), scale=1 / (m + n))
    student_weights = generate_gaussian(key, (m, n), scale=1 / (m + n))
    A_teacher = jnp.array([1, -1])
    A_student = jnp.zeros((2,))
    key, _ = jax.random.split(key)
    config = {}
    config['type'] = 'activity'
    # same random initialization of the weights at the edges for student and teacher network
    if(config['type'] == 'activity'):
        trajectory_loss = activity_trajectory_loss
        generate_trajectory = generate_activity_trajectory
    else:
        trajectory_loss = weight_trajectory_loss
        generate_trajectory = generate_weight_trajectory


    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(A_student)

    expdata = {'A_0':[], 'A_1':[], 'loss':[]} 
    expdata['epoch'] = jnp.arange(meta_epochs)

    for epoch in range(meta_epochs):
        start_time = time.time()
        key = jax.random.PRNGKey(0)
        expdata['loss'].append(0.)

        for _ in range(num_trajectories):
            key, _ = jax.random.split(key)
            x = generate_gaussian(key, (length, m), scale=0.1)
            trajectory = generate_trajectory(
                x, teacher_weights, A_teacher
            )
            # print("weight trajectory", weight_trajectory)
            loss_t, grads = jax.value_and_grad(trajectory_loss, argnums=2)(
                x, student_weights, A_student, trajectory
            )
            expdata['loss'][-1]+=loss_t

            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                A_student,
            )
            A_student = optax.apply_updates(A_student, updates)

        expdata['A_0'].append(A_student[0])
        expdata['A_1'].append(A_student[1])

        print("A student:", A_student)
        print("loss (all trajectories):", expdata['loss'][-1])
        print("epoch {} time: {}s".format(epoch, time.time() - start_time))
        print()

    df = pd.DataFrame(expdata)
    # df.to_csv('exps/scalability_meta_learning/expdata-w-non-linear.csv')


if __name__ == "__main__":
    main()
