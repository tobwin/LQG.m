# LQG

LQG.m implements a copyable handle class for discrete-time, finite-horizon [Linear-Quadratic-Gaussian](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic%E2%80%93Gaussian_control#Discrete_time) estimation and control.

An __LQG object__ represents time-varying system dynamics x<sub>t+1</sub> = A<sub>t</sub>x<sub>t</sub> + B<sub>t</sub>u<sub>t</sub> + v<sub>t</sub> and observations y<sub>t</sub> = C<sub>t</sub>x<sub>t</sub> + w<sub>t</sub> together with a quadratic cost function x<sub>t</sub><sup>T</sup>Q<sub>t</sub>x<sub>t</sub> + u<sub>t</sub><sup>T</sup>R<sub>t</sub>u<sub>t</sub>.
The corresponding [Linear-Quadratic-Regulator](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator#Finite-horizon,_discrete-time_LQR) end [Linear-Quadratic-Estimator](https://en.wikipedia.org/wiki/Kalman_filter#Details) are implemented as dependent object properties and thus computed on the fly. 

Class methods include sampling as well as the computation of various __statistics__, including the __expected cost__ under optimal or custom linear feedback control.

Below we use a toy-model to illustrate class functionality. Further information can be found using the _help LQG_ command in MATLAB. For remaining questions, feedback etc. feel welcome to email winner.tobias@gmail.com


## Example

We will simulate a point-mass subject to gravity on earth. To approximate its continuous second-order dynamics using a discrete-time system, we first define a number of parameters:

```matlab
T = 1; % time horizon in seconds
dt = 0.001; % time increment in seconds (fs = 1 millisecond)
g = 9.81; % gravitational acceleration in m/s²
m = 2; % mass in kg
```

### System Definition

With a constant gravitational acceleration, we need three state dimensions to track the point-mass, namely, its position, velocity and acceleration. We also include a constant 1 for later convenience. The state transition matrix A<sub>t</sub> for this system is:

```matlab
A = [1 dt 0 0
     0 1 dt 0
     0 0 0 -g
     0 0 0 1];
```

An LQG system is initialized with a fixed time-horizon T and state transition matrices A<sub>t</sub>. If A<sub>t</sub> is time-varying, then it must be represented as a cell-array of matrices. In our case A is constant, so we can use the following syntax:

```matlab
lqg = LQG(T/dt, 'A', A);
```

We must initialize the state with the correct acceleration and the constant 1. Let's also intialize the position to be at 1.5m above the ground:

```matlab
lqg.x = [1.5 0 9.81 1]';
```

Let's examine the system's behavior over time. We can compute the systems deterministic trajectory by calling LQG.mean and plotting the position variable:

```matlab
t = linspace(0,T,floor(T/dt));

plot(t,lqg.mean.x(1,:),'LineWidth',2)

ylim([0 2])
xlabel('time (s)')
ylabel('height (m)')
title('free fall')
```

![Free fall](https://raw.githubusercontent.com/tobwin/LQG.m/master/doc/Fig1.png)

### Linear-Quadratic-Regulation

We want to be able and control the point mass by applying forces to it. We can add a corresponding matrix B to the system:

```matlab
lqg.B = [0; 0; 1/m; 0];
```

Next, we specify a quadratic cost function over states and controls. We may want to constantly keep the mass' height at 1.5m, but we also consider a moderate cost of force application. State costs defined a single matrix Q are interpreted as final-state costs. To specify non-final state costs, we need to define Q as a cell-array of matrices:

```matlab
lqg.Q = LQG.repcell(dt*[1 0 0 -1.5]'*[1 0 0 -1.5], T/dt);
lqg.R = dt*5e-5;
```

The corresponding LQR is computed and applied automatically:

```matlab
plot(t,lqg.mean.x(1,:),'r','LineWidth',2)
ylim([0 2])
xlabel('time (s)')
ylabel('height (m)')
title('deterministic control')
```

![Deterministic control](https://raw.githubusercontent.com/tobwin/LQG.m/master/doc/Fig2.png)

Let us next add some noise to the system. All noise covariance matrices must be specified in square root form: For matrix V, the state transition error v will be distrbuted as _N_(0,VV'). Similarly, the initial state will be distributed _N_(x,XX'). We can look at the resulting system trajectories using the LQG.sample function:

```matlab
lqg.X = [.05; 0; 0; 0];
lqg.V = [0; 0; 10/m; 0];

data = lqg.sample(10);
x = LQG.time2trl(data.x);

figure, hold on
for trl = 1:10
    plot(t,x{trl}(1,:),'k')
end
ylim([0 2])
xlabel('time (s)')
ylabel('height (m)')
title('open loop control')
```

![Open loop](https://raw.githubusercontent.com/tobwin/LQG.m/master/doc/Fig3.png)


### Statistics

The sampled data also tell us the cost per simulated trajectory. The exact expected value can be computed using the LQG.value command:

```matlab
>> mean(data.cost.total)

ans =

    0.0255

>> lqg.value

ans =

    0.0267
```

### Linear-Quadratic-Estimation (Kalman filter)

Finally, observing the point-mass' position should allow the controller to stabilize the system. We add a corresponding  matrix C as well as some observation noise W. The Kalman filter together with the state prediction and estimation covariances will again be computed on the fly:

```matlab
lqg.C = [1 0 0 0];
lqg.W = 1;

data = lqg.sample(10);
x = LQG.time2trl(data.x);

figure, hold on
for trl = 1:10
    plot(t,x{trl}(1,:),'k','LineWidth',1)
end
ylim([0 2])
xlabel('time (s)')
ylabel('height (m)')
title('feedback control')
```

![Feedback control](https://raw.githubusercontent.com/tobwin/LQG.m/master/doc/Fig4.png)

We can confirm that the feedback helped lowering the expected cost:

```matlab
>> mean(data.cost.total)

ans =

    0.0204

>> lqg.value

ans =

    0.0211
```
