import math


def projectile(v0, h0, theta, phi, tr, ro, c, area, mass, wind_speed, wind_angle, dt):
    t = 0
    p = 0
    x = [0]
    y = [0]
    z = [h0]

    accel_x = []
    accel_y = []
    accel_z = []

    vx = [v0 * math.sin(math.radians(theta)) * math.cos(math.radians(phi))]
    vy = [v0 * math.sin(math.radians(theta)) * math.sin(math.radians(phi))]
    vz = [v0 * math.cos(math.radians(theta))]

    ground_level = 0
    while z[p] >= ground_level:
        vxx = float(vx[p])
        vyy = float(vy[p])
        vzz = float(vz[p])
        v = math.sqrt(vxx ** 2 + vyy ** 2 + vzz ** 2)

        D = tr * float((ro * area * c) / 2)

        wind_vx = wind_speed * math.cos(math.radians(wind_angle))
        wind_vy = wind_speed * math.sin(math.radians(wind_angle))

        ax = -(D / mass) * (vxx - wind_vx) * v
        ay = -(D / mass) * (vyy - wind_vy) * v
        az = -g - ((D / mass) * vzz * v)

        accel_x.append(ax)
        accel_y.append(ay)
        accel_z.append(az)

        delta_x = (vxx * dt) + (accel_x[p] * (dt ** 2) / 2)
        delta_y = (vyy * dt) + (accel_y[p] * (dt ** 2) / 2)
        delta_z = (vzz * dt) + (accel_z[p] * (dt ** 2) / 2)

        vx.append(vxx + ((accel_x[p]) * dt))
        vy.append(vyy + ((accel_y[p]) * dt))
        vz.append(vzz + ((accel_z[p]) * dt))

        x.append(x[p] + delta_x)
        y.append(y[p] + delta_y)
        z.append(z[p] + delta_z)

        t = t + dt
        p = p + 1

    return x, y, z, t



# Пример использования
g = 9.81  # ускорение свободного падения
c = 0.3  # коэффициент сопротивления среды
dt = 0.01  # шаг времени
v0 = 0.1  # начальная скорость
theta = 120  # угол в градусах (вертикальный угол)
phi = 13  # угол в градусах (горизонтальный угол)
ro = 1.225  # плотность среды
area = 0.1  # площадь поперечного сечения объекта
mass = 1  # масса объекта
tr = True  # учет сопротивления среды
h0 = 30  # начальная высота
wind_speed = 3
wind_angle = 37

x, y, z, t = projectile(v0, h0, theta, phi, tr, ro, c, area, mass, wind_speed, wind_angle, dt)

print("Траектория:")
for i in range(len(x)):
    print(f"t = {i * dt} сек: x = {x[i]} м, y = {y[i]} м, z = {z[i]} м")
print(f"Время полета: {t} сек")
