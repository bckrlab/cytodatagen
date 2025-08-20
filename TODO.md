# TODO

- rework: more OOP oriented approach
    - class for cell populations (represents cell types)
        - should contain information about marker distributions of that cell type
        - need to support independent as well as multivariate distributions
    - class SubjectClass
        - cell populations and their distributions
        - proportions of cell types via dirichlet alpha parameter
        - provides method to generate a new sample of that class
            - sample cell type proportions from dirichlet distribution with class' alpha prior
    - need some method to apply other effects, like noise and batch effects
        - could apply those later, even on combined adata object


- [x] implement class signal effect
- test implementation
- generate data
- [x] upload to gitlab