# Final Project Proposal 

**Qirui Fu**

## A Poisson Equation Solver Based on Monte Carlo Method

### 1. Original Work
###### 1.
Muller, Mervin E. **Some Continuous Monte Carlo Methods for the Dirichlet Problem**. The Annals of Mathematical Statistics 27, no. 3 (1956): 569–89. http://www.jstor.org/stable/2237369.
###### 2.
Rohan Sawhney and Keenan Crane. **Monte Carlo geometry processing: a grid-free approach to PDE-based methods on volumetric domains**. ACM Trans. Graph. 39, 4, Article 123 (August 2020), 18 pages. https://doi.org/10.1145/3386569.3392374
###### 3.
Rohan Sawhney, Dario Seyb, Wojciech Jarosz, and Keenan Crane. **Grid-free Monte Carlo for PDEs with spatially varying coefficients**. ACM Trans. Graph. 41, 4, Article 53 (July 2022), 17 pages. https://doi.org/10.1145/3528223.3530134
###### 4.
Rohan Sawhney, Bailey Miller, Ioannis Gkioulekas, and Keenan Crane. **Walk on Stars: A Grid-Free Monte Carlo Method for PDEs with Neumann Boundary Conditions**. ACM Trans. Graph. 42, 4, Article 80 (August 2023), 20 pages. https://doi.org/10.1145/3592398
###### 5.
Sugimoto Ryusuke. **Toward General-Purpose Monte Carlo PDE Solvers for Graphics Applications**. https://hdl.handle.net/10012/22500
###### 6.
https://www.youtube.com/watch?v=cmgNqCwaPYc

### 2. Summary of Original Work
Poisson Equation is a really important kind of PDE we can meet in many fields. In traditional numerical methods solving Poisson equation, we have to divide the domain into grids or meshes to solve it. However, there are two problems here. Firstly we are always introducing dicretization error in this step. Second, if we have a really complex geometry, it needs so much time and computing resources to discretize it and sometimes we will lose many details. In some extreme cases even correct output is not guaranteed. In these scenarios, Monte Carlo based solver would have an edge because it doesn't need grid or meshes.

In 1956, Muller firstly invented a method called Walk on Sphere(WoS) to solve Poisson equation with Dirichlet boundary condition using Monte Carlo pipeline ([work 1](#1)). In Siggraph 2020, Sawhney and Crane from CMU introduced this method into computer graphics community to handle some geometry processing problems([work 2](#2)). Like other fields, Poisson equation is also common in computer graphics. After that, they researched further in this direction. In Siggraph 2022 they extended this method onto more generalized PDEs([work 3](#3)). In Siggraph 2023 they proposed a new method Walk on Stars(WoSt) to handle Neumann boundary condition of Poisson equation ([work 4](#4)). From that, they developed a complete theoretical framework for solving the Poisson equation via Monte Carlo methods. Based on that, Ryusuke focued on how to apply these methods on computer graphics tasks during his PhD stage([work 5](#5)). Sawhney and Crane they also gave a talk about these works in [6](#6).

### 3. Work in this Project
At first I was interested in simulating fluid with Monte Carlo because we need to solve a huge global Poisson equation in fluid simulation and the treatment of boundary conitions can significantly affect the results. But it's too ambitious and not suitalbe for a Numerical Methods course. So I decided to implement a solver for Poisson equation with Dirichlet boundary condition based on Monte Carlo method. If I have enough time, I will consider two advanced function: Neumann boundary condition and variance reduction. In the end, I will compare the results of my solver with a traditional solver based on Finite Difference. Because it's impractical to run solvers on a really complicated geometry boundary, I plan to test on some geometries like rectangle, circle, and ellipse to verify my solver. 

Schedule:
* **Week 1, 3/23 ~ 3/29:** Read materials mentioned before and understand the algorithm WoS.
* **Week 2&3 3/30 ~ 4/12:** Implement Monte Carlo solver with Dirichlet boundary condition.
* **Week 4 4/13 ~ 4/19** Implement advanced functions, including Neumann boundary condition and variance reduction.
* **Week 5 4/20 ~ 4/26** Implement Finite Difference solver (or download online, I believe I can find some good implementations) and compare the results. Write final report.