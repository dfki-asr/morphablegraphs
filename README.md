# Morphablegraphs

Python implementation of statistical motion modelling and synthesis using Functional Principal Component Analysis and Gaussian Mixture Models. The code is partially based on the paper by Min, Jianyuan, and Jinxiang Chai:  
"Motion graphs++ a compact generative model for semantic motion analysis and synthesis." ACM Transactions on Graphics 31.6 (2012): 1-12.  
  
  
The code uses data structures defined in the [anim_utils](https://github.com/eherr/anim_utils) repository. The [mg_scripts](https://github.com/eherr/mg_scripts) repository provides a command line interface for motion modelling and motion synthesis.  
For the integration with a game engine, the [mg_server](https://github.com/eherr/mg_server) repository contains a stateful motion synthesis server.  
The optional MGRD submodule is a proprietary faster implementation of the core library.
 
## Developers

Han Du<sup>1</sup>, Erik Herrmann<sup>1</sup>, Markus Mauer<sup>2</sup>, Martin Manns<sup>2</sup>, Fabian Rupp<sup>2</sup>  <br/>
  
<sup>1</sup>DFKI GmbH  
<sup>2</sup>Daimler AG  



## License

Copyright (c) 2019 DFKI GmbH, Daimler AG.  
MIT License, see the LICENSE file.  
Contributions of each partner are highlighted in the copyright notice of each file.