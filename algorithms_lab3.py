import math
import matplotlib.pyplot as plt


rho_air = 1.2041
c_drag = 0.47
rho_iron = 7200
d = 0.1
v0 = 1000
angle_degrees = 45
g = 9.81
h = 0.5


def derivatives(state, k):
    x1, x2, v1, v2 = state
    speed = math.sqrt(v1**2 + v2**2)

    dx1 = v1
    dx2 = v2
    dv1 = -k * speed * v1
    dv2 = -k * speed * v2 - g

    return dx1, dx2, dv1, dv2


def add_states(state, change, factor):
    return tuple(
        value + factor * delta
        for value, delta in zip(state, change)
    )


def runge_kutta_step(state, h, k):
    k1 = tuple(h * value for value in derivatives(state, k))

    state2 = add_states(state, k1, 0.5)
    k2 = tuple(h * value for value in derivatives(state2, k))

    state3 = add_states(state, k2, 0.5)
    k3 = tuple(h * value for value in derivatives(state3, k))

    state4 = add_states(state, k3, 1)
    k4 = tuple(h * value for value in derivatives(state4, k))

    new_state = tuple(
        value + (a + 2 * b + 2 * c + d) / 6
        for value, a, b, c, d in zip(state, k1, k2, k3, k4)
    )

    return new_state


def calculate_trajectory(k):
    angle = math.radians(angle_degrees)

    state = (
        0,
        0,
        v0 * math.cos(angle),
        v0 * math.sin(angle)
    )

    times = [0]
    states = [state]

    while True:
        next_state = runge_kutta_step(state, h, k)

        if next_state[1] <= 0:
            fraction = -state[1] / (next_state[1] - state[1])

            landing_state = tuple(
                old + fraction * (new - old)
                for old, new in zip(state, next_state)
            )

            times.append(times[-1] + fraction * h)
            states.append(landing_state)
            break

        times.append(times[-1] + h)
        states.append(next_state)
        state = next_state

    return times, states


radius = d / 2
area = math.pi * radius**2
mass = rho_iron * (4 / 3) * math.pi * radius**3
k_with_air = c_drag * rho_air * area / (2 * mass)

cases = [
    ("Без сопротивления воздуха", 0),
    ("С сопротивлением воздуха", k_with_air)
]

results = []

for name, k in cases:
    times, states = calculate_trajectory(k)

    x1 = [state[0] for state in states]
    x2 = [state[1] for state in states]

    flight_time = times[-1]
    distance = x1[-1]
    max_height = max(x2)

    print(name)
    print(f"k = {k:.8f} 1/м")
    print(f"Время полёта: {flight_time:.2f} с")
    print(f"Дальность полёта: {distance:.2f} м")
    print(f"Максимальная высота: {max_height:.2f} м")
    print()

    results.append((name, times, x1, x2))


print(f"Масса снаряда: {mass:.4f} кг")
print(f"Площадь сечения: {area:.6f} м^2")


fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for row, (name, times, x1, x2) in enumerate(results):
    axes[row, 0].plot(times, x1)
    axes[row, 0].set_title(f"{name}: x1(t)")
    axes[row, 0].set_xlabel("Время t, с")
    axes[row, 0].set_ylabel("Координата x1, м")
    axes[row, 0].grid()

    axes[row, 1].plot(times, x2)
    axes[row, 1].set_title(f"{name}: x2(t)")
    axes[row, 1].set_xlabel("Время t, с")
    axes[row, 1].set_ylabel("Высота x2, м")
    axes[row, 1].grid()

    axes[row, 2].plot(x1, x2)
    axes[row, 2].set_title(f"{name}: x2(x1)")
    axes[row, 2].set_xlabel("Координата x1, м")
    axes[row, 2].set_ylabel("Высота x2, м")
    axes[row, 2].grid()

plt.tight_layout()
plt.show()