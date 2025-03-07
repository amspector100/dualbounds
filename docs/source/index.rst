.. image:: images/logo.svg
  :width: 100%
  :alt: Horizontal logo for dualbounds.

Introduction
============

``dualbounds`` is a python framework for performing inference on partially identified causal estimands (and certain identified estimands as well).

In randomized experiments, dual bounds can wrap around any machine learning algorithm to provide provably valid inference, even if the underlying ML model is misspecified or inconsistent. When applied to observational data, dual bounds also have strong double-robustness guarantees.

Citation
========

If you ``dualbounds`` in an academic publication, please consider citing our paper:

Bibtex entry::

 @misc{ji_lei_spector_2023,
      title={
         Model-Agnostic Covariate-Assisted Inference 
         on Partially Identified Causal Effects
      }, 
      author={Wenlong Ji and Lihua Lei and Asher Spector},
      year={2023},
      eprint={2310.08115},
      archivePrefix={arXiv},
      primaryClass={econ.EM},
      url={https://arxiv.org/abs/2310.08115},
 }


.. toctree::
   :caption: Installation
   :hidden:

   installation.rst

.. toctree::
   :caption: Quickstart
   :hidden:

   quickstart.ipynb

.. toctree::
   :caption: User Guide
   :hidden:

   tutorials/math_review.ipynb
   tutorials/dualbounds_class.ipynb
   tutorials/deltadualbounds.ipynb
   tutorials/additional_estimands.ipynb
   tutorials/model_selection.ipynb
   tutorials/support_restrictions.ipynb
   tutorials/diagnostics.ipynb

.. toctree::
   :caption: Troubleshooting
   :hidden:

   troubleshooting.rst

.. toctree::
   :caption: API Reference
   :hidden:

   apiref.rst

.. toctree::
   :caption: arXiv
   :hidden:
   
   https://arxiv.org/abs/2310.08115

Index
=====

* :ref:`genindex`
* :ref:`modindex`

Thanks
======

Thanks to Kevin Guo, whose comments substantially improved the API structure.