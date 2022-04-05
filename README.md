# Morphablegraphs

Python implementation of statistical motion modelling and synthesis using Functional Principal Component Analysis and Gaussian Mixture Models. The code is partially based on the paper by Min, Jianyuan, and Jinxiang Chai:  
"Motion graphs++ a compact generative model for semantic motion analysis and synthesis." ACM Transactions on Graphics 31.6 (2012): 1-12.  

The code uses data structures defined in the [anim_utils](https://github.com/eherr/anim_utils) repository. 
The optional [MGRD](https://github.com/dfki-asr/mgrd) library is a proprietary faster implementation of the core.


## Usage

The examples directory provides a command line interface for motion modelling and motion synthesis. Note, the [mg_server](https://github.com/eherr/mg_server) repository provides a stateful  motion synthesis server for integration with a game engine. The motion modelling and the motion synthesis can also be run inside the [motion_preprocessing_tool](https://github.com/eherr/motion_preprocessing_tool).


### Motion Primitive Construction

Call the motion primitive modelling pipeline mg_construction_pipeline.py from the command line:
```bat  
python examples/run_construction.py --name walk  --skel_filename skeleton.bvh --input_folder data/walk --output_folder out
```

### Motion Synthesis

For the integration with a game engine, the [mg_server](https://github.com/eherr/mg_server) repository contains a stateful motion synthesis server.  
The script mg_rest_interface.py provides a legacy REST interface localhost:port/run_morphablegraphs to rus the synthesis offline.
It can be called using an POST message with a string containing the constraints as message body. An example input constraints is provided in example/example_input.json.


## Developers

Han Du<sup>1</sup>, Erik Herrmann<sup>1</sup>, Markus Mauer<sup>2</sup>, Martin Manns<sup>2</sup>, Fabian Rupp<sup>2</sup>  <br/>

<sup>1</sup>DFKI GmbH  
<sup>2</sup>Daimler AG  



## License

Copyright (c) 2019 DFKI GmbH, Daimler AG.  
MIT License, see the LICENSE file.  
Contributions of each partner are highlighted in the copyright notice of each file.