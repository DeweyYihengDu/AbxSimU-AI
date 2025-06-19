# DL_on_antibiotics_microbiology
 
## Source Data

The dataset of current study comprises three distinct libraries of compounds, adopted from a former study \autocite{swanson_2024_generative}. 

 - **Library 1** consists of 2371 internationally approved drug compounds from the Pharmakon-1760 library and 800 natural compounds isolated from plant, animal, and microbial sources. Pharmakon-1760 is a curated chemical library developed by MicroSource Discovery Systems, containing 1360 Food and Drug Administration (FDA)-approved drugs and 400 internationally approved drugs. This library is widely used in high-throughput drug screening and drug repurposing studies due to its inclusion of clinically relevant molecules with diverse biological activities.
 - **Library 2**, known as the Broad Drug Repurposing Hub, contains 6680 molecules, many of which are FDA-approved drugs or candidates currently undergoing clinical trials. 
 - **Library 3** is a synthetic small-molecule screening collection comprising 5376 molecules randomly sampled from a larger chemical library maintained at the Broad Institute. All three libraries were screened for their growth inhibitory activities against an opportunistic bacteria, namely - *Acinetobacter baumannii*, primarily associated with hospital-acquired infections and resistance. The activities were categorized as active (1) or inactive (0), constituting a binary classification problem.

In our research, we compared two modeling strategies: (1) a Combined libraries approach, where Libraries 1, 2, and 3 were merged prior to data splitting; and (2) a Library partitioning approach, in which Libraries 1 and 2 were used exclusively for training, and Library 3 was reserved for independent testing.