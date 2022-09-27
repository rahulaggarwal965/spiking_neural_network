#### General Outline

* Say we have a neuron *i*
* For each sequence of spikes F<sub>*0*</sub>, F<sub>*1*</sub>, ... F<sub>*k*</sub> from input neurons *0, 1, ... k*, multiply them by a weight vector W<sub>*ki*</sub> and add them to neuron *i*'s "state". 
    * Each spike will generate a **PSP** determiined by the equation: PSP(*t*) = *e*<sup>(*-t/τ<sub>m</sub>*)</sup> - *e*<sup>(*-t/τ<sub>s</sub>*)</sup>
    * Add each of these PSP's to neuron *i's* state and model neuron 
    * Akin to integrating over the spikes

* Neuron *i* now has a series of states S<sub>t</sub> from t = t<sub>*0*</sub> to t
* Decay that state every time step
* Threshold? Generate spike at time t
* Enter refractory period *η*(*t*)=*−ve*<sup>*(t/τ<sub>r</sub>)*</sup>*H(t)*


