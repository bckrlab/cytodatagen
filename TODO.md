# TODO

## 2025-09-24

- cytodatagen.make_subjects --n-subjects 2 --n-signal-markers 2
    - implemented by class SubjectGenerator
    - checks if n-markers or marker names is given
    - checks if n-subjects or subject names is given
    - checks if n-pop or pop_names is given
    - creates a new control subject
    - creates n-1 signal subjects based on that control subject
    - writes subjects to csv
- cytodatagen --subjects subjects.json --n-cells 1000
    - loads subjects from json
        - restores subject names
        - restores populations


- TODOS
    - cleanup code, check what is actually needed
    - decouple SignalPopulation, so that it doesn't depend anymore on a ControlPopulation
        - makes it easier to serialize / deserialize as json
    - implement to_dict and from_dict methods for serialization to/from json
        - classes: subject, populations, marker_dist, distributions
        - subject needs a population registry for deserialization to find the right population to pass the args to
        - NamedMarkerDist / MarkerDistribution needs a Distribution registry
        - then, we should be good to go and can just call subject.to_dict() and subject.from_dict()

## 2025-09-08

- objective: generate synthetic fluoroessence signals for flow cytometry data
- user provides a config file which defines:
    - number of cell types
    - number of subjects
    - number of signal cell types (will have shifted distributions)
    - number of signal markers (will have shifted distributions)
- actually, can just provide another config to the sampler
    - defines subjects with alpha values, cell type names and markers
    - defines number of cells to sample for each subject
    - fixes random seed
- problems with current interface:
    - we used ControlPopulation and SignalPopulation classes - hard to serialize, as SignalPopulation constructor expects a ControlPopulation object
        - instead, would be easier if config just defines mv normal distributions with mean and 
        - main problem as usual: proper way in Python to serialize to and reconstruct objects from a config file
    - want user to be able to pre-configure alpha values and signal cell types / signal markers
        - seems a bit ugly, to have both n_ct and ct_names around
- classes:
    - population: represents one cell type with their associated marker distribution
    - subject: represents one test subject / class, defines the subjects cell types and their proportions (dirichlet alpha values)
- config.py
    - 

## 2025-09-02

- want to write a config generator
    - should assist user in generating the config.json for the actual data generation
        - specifies subjects, i.e., their marker distributions and alphas
        - specifies transforms
        - specifies other parameters like random seed
        - then, user can just go `cytodatagen --config config.json -o cytodata/`
    - problem: multiple steps, where we want user to be able to either specify their own values, or generate random ones
        - for example: ct_names, marker_names, subject_names, alpha values for each subject, signal ct and signal markers
        - could provide option to pass a `preconfig.json`, and script just generates missing values
    - problem: structure is a little bit different:
        - i.e., if I have all subjects fully specified, I don't need ct_names, marker_names, alphas, etc. anymore
    - basically, we have a series of steps, where we have either:
        - value provided: taken as is, maybe run checks
        - generate: then, user needs to specify parameters for generation

## 2025-09-01

- need to construct control subject and signal subjects
    - can first create control subject
    - each signal subject is constructed from a control subject
    - each need a different alpha vector
        - want to be able to either specify this vector explicitly
        - or generate it via a SignalAlphaBuilder or something like that
    - 
- then, we can use the control subjects and the signal subjects to construct the generator
    - this should be straight forward
    - only complication might be to distinguish between using provided parameters, and generating them
    - more straightforward: always expect explicit parameters, but provide script to generate them
- finally, want to be able to export generated parameters, to rebuild dataset one to one
    - here, it is definitely necessary to separate parameter generation and data generation
    - otherwise, our rng gets messed up if we either use or not use it to generate parameters
    - actually, we could avoid this by resetting the random seed after parameter generation is complete


##

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