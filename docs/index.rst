.. jazz-jitter-analysis documentation master file, created by
   sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Code for: Leader-Follower Relationships Optimize Coordination in Networked Jazz Performances
============================================================================================

.. image:: https://user-images.githubusercontent.com/97224401/231712093-133cafa0-dffe-4a23-945d-5249c4385bab.gif
   :alt: StreamPlayer
   :align: center

This repository is associated with the paper “Leader-Follower Relationships Optimize Coordination in Networked Jazz Performances”, published in JOURNAL, and includes scripts for reproducing the analysis, data, and figures contained in this paper. The corresponding dataset, comprising audio and video recordings of 130 individual jazz duo performances, biometric data, and subjective evaluations and comments from both the musicians and listeners recruited in a separate perceptual study, is `freely accessible on Zenodo <https://doi.org/10.5281/zenodo.7773824>`_

Quick start
-----------

To build the dataset, models, simulations, figures, and audio-visual stimuli used in the original paper, follow the instructions in :ref:`Getting started`.

After building, for examples on how to use the dataset in your own Python sessions, take a look at :ref:`Examples`.

For help, including advice on building the dataset on slower systems, see :ref:`Troubleshooting`.

For the source code documentation (not comprehensive or pretty), go to :ref:`modindex`

Finally, to explore the models and data created here without having to install or run anything locally, check out :ref:`Notebooks`.

Requirements
------------

- Windows 10 (other operating systems including Ubuntu may work but are not tested)
- Python 3.10+
- Git
- FFmpeg (**only for reproducing combined audio-visual stimuli**)

Contents:
---------

.. toctree::
   :maxdepth: 2

   getting-started
   examples
   troubleshooting
   notebooks

Documentation:
--------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   src

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
