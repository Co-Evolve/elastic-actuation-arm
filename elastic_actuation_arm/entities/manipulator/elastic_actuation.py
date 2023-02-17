import numpy as np
from dm_control.mjcf import Physics

from elastic_actuation_arm.entities.manipulator.specification import ParallelSpringSpecification, \
    ManipulatorMorphologySpecification, BiarticularSpringSpecification


def apply_elastic_actuation(physics: Physics,
                            shoulder_joint, elbow_joint,
                            spec: ManipulatorMorphologySpecification) -> None:
    shoulder_physics = physics.bind(shoulder_joint)
    elbow_physics = physics.bind(elbow_joint)

    parallel_elastic_actuation(joint_physics=shoulder_physics, spring_spec=spec.upper_arm_spec.spring_spec,
                               counter_clockwise=True)
    parallel_elastic_actuation(joint_physics=elbow_physics, spring_spec=spec.fore_arm_spec.spring_spec,
                               counter_clockwise=False)
    biarticular_elastic_actuation(shoulder_physics=shoulder_physics,
                                  elbow_physics=elbow_physics,
                                  spring_spec=spec.biarticular_spring_spec)


def parallel_elastic_actuation(joint_physics: Physics, spring_spec: ParallelSpringSpecification,
                               counter_clockwise: bool) -> None:
    stiffness = spring_spec.stiffness.value
    equilibrium_angle = spring_spec.equilibrium_angle.value

    current_angle = joint_physics.qpos[0]

    if counter_clockwise:
        should_apply_force = current_angle <= equilibrium_angle
    else:
        should_apply_force = current_angle >= equilibrium_angle

    if should_apply_force:
        joint_physics.qfrc_passive += stiffness * (equilibrium_angle - current_angle)


def biarticular_elastic_actuation(shoulder_physics, elbow_physics, spring_spec: BiarticularSpringSpecification) -> None:
    q0 = spring_spec.q0.value
    q2 = shoulder_physics.qpos[0]
    # negate the elbow's position because we work in a frame where positive should point counter-clockwise
    q3 = -elbow_physics.qpos[0]

    # Add offsets to incorporate the diameter of the cables
    r1 = spring_spec.r1.value + 0.001
    r2 = spring_spec.r2.value + 0.001
    r3 = spring_spec.r3.value + 0.001

    x = (-r2 * q2 - r3 * q3 - r1 * q0)

    if x > 0:
        k = spring_spec.stiffness.value / (r1 ** 2)

        shoulder_torque = k * r2 * x
        # Biarticular spring applies torque that counteracts gravity -> negate elbow torque
        elbow_torque = -(k * r3 * x)
        shoulder_physics.qfrc_passive += shoulder_torque
        elbow_physics.qfrc_passive += elbow_torque


def calculate_load_torque(joint_index: int, torque: np.ndarray, vel: np.ndarray, dt: float = None,
                          acc: np.ndarray = None) -> float:
    # Constants (supplied by Maxime)
    torque_to_current = [10.89, 12.34, 10.89]

    c_0 = [0.5669, 0.7080, 0.5085]
    nu = [2.5853, 3.3084, 2.1345]
    c_l = [0.098, 0.082, 0.098]
    J_m = [0, 0.0559, 0.0034]
    c_v2 = [-0.7760, -1.2718, -0.5508]
    c_l2 = [-0.014, 0.011, 0.005]
    b = 20

    current = torque / torque_to_current[joint_index]
    if acc is None:
        assert dt is not None, "calculate_load_torque requires dt or acc to be given"
        acc = np.gradient(vel, dt)

    if joint_index == 2:
        nom = current - (c_0[joint_index] * np.tanh(b * vel) + nu[joint_index] * vel + J_m[joint_index] * acc +
                         c_v2[joint_index] * np.tanh(b * vel) * vel ** 2)
        denom = c_l[joint_index] + c_l2[joint_index] * np.tanh(b * -vel)
        return -nom / denom
    else:
        nom = current - (c_0[joint_index] * np.tanh(b * vel) + nu[joint_index] * vel + J_m[joint_index] * acc +
                         c_v2[joint_index] * np.tanh(b * vel) * vel ** 2)
        denom = c_l[joint_index] + c_l2[joint_index] * np.tanh(b * vel)
        return nom / denom
