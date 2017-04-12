# Embedding and GRU benchmark

## PyTorch

### Without packing

```
$ python uniskip.py 
Setup : compile + forward/backward x 1
--- 6.833168268203735 seconds ---
Forward:
--- 10000 samples in 2.982395648956299 seconds (3353.0148125521573 samples/s, 298.23906421661377 microsec/samples) ---
Forward + Backward:
--- 10000 samples in 13.867890357971191 seconds (721.090336098986 samples/s, 1386.78879737854 microsec/samples) ---
```

### With packing

```
$ python uniskip_pack.py 
Setup : compile + forward/backward x 1
--- 6.408475160598755 seconds ---
Forward:
--- 10000 samples in 3.060577154159546 seconds (3267.3637004753537 samples/s, 306.05714321136475 microsec/samples) ---
Forward + Backward:
--- 10000 samples in 16.286936044692993 seconds (613.989151917149 samples/s, 1628.6932706832886 microsec/samples) ---
```

## Torch7

### Without trimZero

```
$ th uniskip.lua
Setup : compile + forward/backward x 1 
--- 13.117871999741 seconds ---  
Forward: 
--- 10000 samples in 5.9476850032806 seconds (1681.3278464713 samples/s, 594.76799964905 microsec/samples) --- 
Forward + Backward:  
--- 10000 samples in 16.117718219757 seconds (620.43537294494 samples/s, 1611.7714166641 microsec/samples) --- 
```

### With trimZero

```
$ th uniskip.lua
Setup : compile + forward/backward x 1 
--- 13.256095170975 seconds ---  
Forward: 
--- 10000 samples in 6.7783710956573 seconds (1475.2815740486 samples/s, 677.83670425415 microsec/samples) --- 
Forward + Backward:  
--- 10000 samples in 18.175946950912 seconds (550.17781365844 samples/s, 1817.5941944122 microsec/samples) ---
``` 

### With trimZero and 70% of the sequence is null

```
$ th uniskip.lua
Setup : compile + forward/backward x 1 
--- 11.403578996658 seconds ---  
Forward: 
--- 10000 samples in 2.9037292003632 seconds (3443.8532496074 samples/s, 290.3724193573 microsec/samples) ---  
Forward + Backward:  
--- 10000 samples in 10.128414869308 seconds (987.32169795254 samples/s, 1012.8411054611 microsec/samples) ---
``` 