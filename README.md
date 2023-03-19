# Finding Periodic Discrete Events in Noisy Streams

This project is an informal implementation of a particle filter applied to detecting discrete time events in noisy signals, proposed by `(Ghosh, Lucas and Sarkar, 2017)`. I refer the reader to the original authors' abstract for a brief summary of this model.


> Periodic phenomena are ubiquitous, but detecting and predicting periodic events can be difficult in noisy environments. 
> We describe a model of periodic events that covers both idealized and realistic scenarios characterized by multiple kinds of noise.
> The model incorporates false-positive events and the possibility that the underlying period and phase of the events change over time. 
> We then describe a particle filter that can efficiently and accurately estimate the parameters of the process generating periodic events 
> intermingled with independent noise events.

> The system has a small memory footprint, and, unlike alternative methods, its computational complexity is constant in the number of events 
> that have been observed. As a result, it can be applied in low-resource settings that require real-time performance over long periods of time. 
> In experiments on real and simulated data we find that it outperforms existing methods in accuracy and can track changes in periodicity and 
> other characteristics in dynamic event streams.

---

# Quick Install
As this project is intended to be an informal implementation, I would suggest using an editable installation of this package so that the user can refer back to the source code and make custom changes they wish to make.

```
git clone https://github.com/bl3e967/PeriodicEventDetection.git

cd <path-to-repo>

python -m pip install -e .
```

# Usage

```
conf = Config(
  signal=signal_timestamps, 
  num_particles=256, 
)

model = PeriodicityParticleFilter(conf)

res = model.fit()
```

Example usage can be found in `notebooks/research.ipynb`. 

# Notes

During testing, it was found that the model performance is sensitive to the scale parameters for the hypothesis update step of the particle filter. An improper scale means the respective parameter never converges, or diverges after converging. Therefore, the user should calibrate these parameters for their specific application. 

It was also found that $\sigma$ does not converge to the actual value, whereas all other parameters converged. This was the case even when the noise component of the input signal was significantly reduced. Future iterations of this project will need to delve deeper into why this is. 

I refer the reader to the jupyter notebook detailing the research I undertook while writing this code. This can be found in `./notebooks/Research.ipynb`. 

# Reference
<a id="1">[1]</a> Ghosh, Lucas and Sarkar (2017). Finding Periodic Discrete Events in Noisy Streams.
```
@inproceedings{inproceedings,
author = {Ghosh, Abhirup and Lucas, Christopher and Sarkar, Rik},
year = {2017},
month = {11},
pages = {627-636},
title = {Finding Periodic Discrete Events in Noisy Streams},
doi = {10.1145/3132847.3132981}
}```

