Problems with industrial optimization

1. Problem definition changes
2. Business constraints
3. Large scale and combinatorial
4. Short running times

Examples: Audience forecast (TF1), Telecom user behavior, Failure prediction.
Comparing with academics optimization: linear regression, SVM, etc.

Operations research:
1. Minimize the cost of optimization process along with near-optimal solution
   
   a. Network design

   b. Maintenance planning


   Difficulties:
    - Lots of mathematical properties and algorithms
    - Tractability: non-convexity, integrity, black box
    - Scalability
2. Algorithmic Process
   1. Create model
      1. Understanding the problem
      2. Find decisions/actions
      3. Model the objective
      4. Find the constraints
   2. Solve
      1. Good solution, good bound, optimal solution
      2. Engineering: exact or heuristic
    
    Example:
        
        Problem: package delivery
        Objective: minimize CO2, total distance

        Insights: remove crossing edges iteratively

3. Business Process: modeling and then using solvers

