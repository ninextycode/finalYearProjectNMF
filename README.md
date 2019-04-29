# Final Year Project
## Nonnegative Matrix Factrization(NMF)


### The source code structure:

* nmf - CPU nonnegative matrix factorization algorithms
  * bayes.py - Bayesian nonnegative matrix factorization
  * mult.py - multiplicative nonnegative matrix factorization
  * nesterov.py - NMF based on Nesterov gradient descent 
  * norms.py - Frobenius norm and  Kullback–Leibler divergence measure
  * pgrad.py - NMF based on the projected gradient descent 

* nmf_torch - GPU nonnegative matrix factorization algorithms
  * \* analogous to the nmf folder *

* performance
  * performance_eval_func - functions used for performance evaluation and comparison

* read_data
  * reading.py - functions to read downloaded real-life data into memory

* theory - Scripts related to the theoretical part of the project. Focus on the connection between NMF and finding the roots of a polynomial 
  * fact_of_transform.py - for each polynomial equation, a corresponding nonnegative matrix can be constructed. The functions of this file can be used to create an exact nonnegative factorisation of the particular matrix, given the solution to the corresponding polynomial(solution restricted to [0, 1] interval)
  * poly_solution_from_fact.py - The functions of this file can be used to find a root of the polynomial given the factorization of its corresponding nonnegative matrix
  * represent.py - functions which can transform nonnegative factorisation presented as a sum of rank 1 matrices into the form of a product of 2 matrices and vice versa 
  * transform.py - functions which can, given a polynomial, create a nonnegative matrix whose nonnegative factorisation is equivalent to finding the roots of the polynomial restricted to [0, 1] interval
  * visual.py - some visualisation functions

* main.py, main_torch.py, - rough work and testing
* formula_factorisation.ipynb - demonstration of the scripts related to the theoretical part of the project
* performance_on_face_data.ipynb - performance comparison on the facial image dataset
* performance_on_hyperspectral_im.ipynb - performance comparison on the hyperspectral images
* performance_on_random.ipynb - performance comparison on the random data
* performance_on_text_data.ipynb - performance comparison on the text dataset
* algorithms_demo.ipynb - demonstration of the approximate algorithms 


### Notes on running the code:

The best way to run the code is to use the Jupyter Notebook. Notebook demonstration of the scripts related to the theoretical part of the project can be easily run. However, notebooks which focus on the performance comparison bay take a long time to complete, require the presence of GPU and require to download external data. To see the approximate NMF algorithms working, it is suggested to consider algorithms_demo.ipynb notebook instead of the notebooks which focus on the performance comparison. 

### Real-life data which was used for performance analysis:

To obtain the datasets which were used in the scope of the project, enter the "data" folder, open the terminal and run script "downlodad_data.sh". The script was tested on Ubuntu 16.04. To run this script successfully, you must have sufficient available storage. In addition, you must have the permissions to install package "mmv - Move/Copy/Append/Link multiple files" which will be used to rearrange downloaded data. Installation included in the downloading script.

### External libraries

All the external libraries are listed in requirements.txt
