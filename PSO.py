import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Inner_feedback_2 as In2
import clustering.DBSCAN as db
from clustering.CLSC import CLSC


def pso(data, eps_SCL, eps_GCL, clusters, num, core_data, params):
    # Parameter settings
    N = 30  # Number of particles
    omega_max = 1.8
    omega_min = 0.3
    c1_init = 2.5
    c1_final = 0.1
    c2_init = 3
    c2_final = 2.25
    itermax = 60
    tolerance = 0.001

    # Initialization
    global_best_scores_per_iteration = []

    # Initialize particle positions and velocities
    particles_pos = np.random.rand(N, 3)  # Each particle has 3 dimensions: eps_LL, eps_HL, eps_LH
    particles_vel = (np.random.rand(N, 3) - 0.5) * 2  # Initialize velocities with wider range
    particles_best_pos = particles_pos.copy()
    particles_best_score = np.full(N, float('inf'))  # Initialize with infinity
    global_best_pos = None
    global_best_score = float('inf')

    # Linear decreasing inertia weight
    def inertia_weight(t):
        return omega_max - (omega_max - omega_min) * t / itermax

    # Individual and social learning factors
    c1 = c1_init
    c2 = c2_init
    decrement_c1 = (c1_init - c1_final) / itermax
    decrement_c2 = (c2_init - c2_final) / itermax

    # Fitness function
    def fitness_function(data, clusters, num, core_data, params, eps_SCL, eps_GCL, eps_LL, eps_HL, eps_LH):
        objective_function_value = In2.inner_feedback_2(data, clusters, num, core_data, params, eps_SCL, eps_GCL, eps_LL, eps_HL, eps_LH)
        return objective_function_value

    # Initialize global best position and score (before PSO loop)
    global_best_pos_initialized = False
    for i in range(N):
        score = fitness_function(data, clusters, num, core_data, params, eps_SCL, eps_GCL, particles_pos[i, 0], particles_pos[i, 1], particles_pos[i, 2])
        if not global_best_pos_initialized or score < global_best_score:
            global_best_score = score
            global_best_pos = particles_pos[i].copy()
            global_best_pos_initialized = True

    # Main PSO loop
    for t in range(itermax):
        # Update inertia weight and learning factors
        omega = inertia_weight(t)
        c1 -= decrement_c1
        c2 -= decrement_c2

        # Update velocities and positions
        for i in range(N):
            r1, r2 = np.random.rand(3), np.random.rand(3)
            cognitive_component = c1 * r1 * (particles_best_pos[i] - particles_pos[i])
            social_component = c2 * r2 * (global_best_pos - particles_pos[i])
            particles_vel[i] = omega * particles_vel[i] + cognitive_component + social_component
            particles_pos[i] += particles_vel[i]

            # Ensure positions stay within reasonable bounds (0 to 1)
            particles_pos[i] = np.clip(particles_pos[i], 0.1, 1)

        # Calculate fitness
        min_score_this_iteration = float('inf')
        for i in range(N):
            score = fitness_function(data, clusters, num, core_data, params, eps_SCL, eps_GCL, particles_pos[i, 0], particles_pos[i, 1], particles_pos[i, 2])
            if score < particles_best_score[i]:
                particles_best_score[i] = score
                particles_best_pos[i] = particles_pos[i].copy()

        # Update global best score and position (optimized logic)
        global_best_idx = np.argmin(particles_best_score)
        current_global_best_score = particles_best_score[global_best_idx]
        if current_global_best_score < global_best_score:
            global_best_score = current_global_best_score
            global_best_pos = particles_best_pos[global_best_idx].copy()

        # Record global best score for each iteration
        global_best_scores_per_iteration.append(global_best_score)

        # Check stopping condition
        if t > 0 and abs(global_best_score - particles_best_score[np.argmin(particles_best_score)]) < tolerance:
            print('Converged')
            break

    # Output results
    print("global_best_score:", global_best_score)
    print("Optimal eps_LL:", global_best_pos[0])
    print("Optimal eps_HL:", global_best_pos[1])
    print("Optimal eps_LH:", global_best_pos[2])
    print('\n')

    # Plot line chart using global best scores per iteration
    plt.plot(range(1, len(global_best_scores_per_iteration) + 1), global_best_scores_per_iteration, marker='o')
    plt.xlabel('Number of iterations')
    plt.ylabel('Adjustment cost')
    plt.title('Iteration result')
    plt.grid(True)
    # plt.show()

    return global_best_pos[0], global_best_pos[1], global_best_pos[2]