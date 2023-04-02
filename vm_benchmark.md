### VM Benchmark
Computing the [79,867 node](data/79867.txt) dataset.  
Operating on AWS EC2: `c6i.8xlarge` (3rd generation Intel Xeon Scalable processors).

#### Results
Finishes in 12.914 seconds on the `c6i.8xlarge` VM.  
On my personal laptop (`i7 8750H` CPU), it finishes in about ~58 seconds.

See the [exported solution](soln/79867.soln).

```sh
root@19bb3cef6106:/subsetlatticebuilder# target/release/subset-lattice-builder "data/79867.txt" "soln/79867.soln"
Layer-1 done.
        - Progress: 0.00%
        - Current duration: 0.002s
        - ETA: 2842.0106638449547s
Layer-2 done.
        - Progress: 0.00%
        - Current duration: 0.002s
        - ETA: 268.4475170440199s
Layer-3 done.
        - Progress: 0.00%
        - Current duration: 0.003s
        - ETA: 78.09797856137129s
Layer-4 done.
        - Progress: 0.03%
        - Current duration: 0.005s
        - ETA: 19.89477708397286s
Layer-5 done.
        - Progress: 0.11%
        - Current duration: 0.009s
        - ETA: 8.555934125883326s
Layer-6 done.
        - Progress: 0.35%
        - Current duration: 0.018s
        - ETA: 5.0538832533724305s
Layer-7 done.
        - Progress: 0.54%
        - Current duration: 0.025s
        - ETA: 4.567114888715263s
Layer-8 done.
        - Progress: 0.64%
        - Current duration: 0.028s
        - ETA: 4.350343365919438s
Layer-9 done.
        - Progress: 0.69%
        - Current duration: 0.03s
        - ETA: 4.301272576501763s
Layer-10 done.
        - Progress: 0.74%
        - Current duration: 0.032s
        - ETA: 4.291701463151073s
Layer-11 done.
        - Progress: 0.81%
        - Current duration: 0.034s
        - ETA: 4.1835971058689205s
Layer-12 done.
        - Progress: 0.88%
        - Current duration: 0.036s
        - ETA: 4.049762307484018s
Layer-13 done.
        - Progress: 0.96%
        - Current duration: 0.039s
        - ETA: 4.018763220935297s
Layer-14 done.
        - Progress: 1.04%
        - Current duration: 0.043s
        - ETA: 4.083474725878008s
Layer-15 done.
        - Progress: 1.13%
        - Current duration: 0.047s
        - ETA: 4.101617969189007s
Layer-16 done.
        - Progress: 1.27%
        - Current duration: 0.053s
        - ETA: 4.128276539407112s
Layer-17 done.
        - Progress: 1.50%
        - Current duration: 0.066s
        - ETA: 4.326625419655461s
Layer-18 done.
        - Progress: 1.94%
        - Current duration: 0.092s
        - ETA: 4.658228618428894s
Layer-19 done.
        - Progress: 2.60%
        - Current duration: 0.136s
        - ETA: 5.100727148784257s
Layer-20 done.
        - Progress: 3.46%
        - Current duration: 0.198s
        - ETA: 5.531719899001335s
Layer-21 done.
        - Progress: 4.39%
        - Current duration: 0.267s
        - ETA: 5.816607240044558s
Layer-22 done.
        - Progress: 5.47%
        - Current duration: 0.346s
        - ETA: 5.977754214145665s
Layer-23 done.
        - Progress: 6.98%
        - Current duration: 0.45s
        - ETA: 5.998750229160038s
Layer-24 done.
        - Progress: 8.86%
        - Current duration: 0.58s
        - ETA: 5.962593103448275s
Layer-25 done.
        - Progress: 11.22%
        - Current duration: 0.725s
        - ETA: 5.7346470302237424s
Layer-26 done.
        - Progress: 14.84%
        - Current duration: 0.914s
        - ETA: 5.244359684396259s
Layer-27 done.
        - Progress: 19.30%
        - Current duration: 1.162s
        - ETA: 4.857497762026875s
Layer-28 done.
        - Progress: 24.54%
        - Current duration: 1.466s
        - ETA: 4.507286463227817s
Layer-29 done.
        - Progress: 30.24%
        - Current duration: 1.807s
        - ETA: 4.169111094760739s
Layer-30 done.
        - Progress: 37.60%
        - Current duration: 2.217s
        - ETA: 3.6785856394312684s
Layer-31 done.
        - Progress: 47.07%
        - Current duration: 2.806s
        - ETA: 3.1556609653467875s
Layer-32 done.
        - Progress: 57.88%
        - Current duration: 3.598s
        - ETA: 2.6178392293837645s
Layer-33 done.
        - Progress: 68.86%
        - Current duration: 4.553s
        - ETA: 2.058967196819169s
Layer-34 done.
        - Progress: 78.69%
        - Current duration: 5.661s
        - ETA: 1.532608605999818s
Layer-35 done.
        - Progress: 86.58%
        - Current duration: 7.072s
        - ETA: 1.0965805185002653s
Layer-36 done.
        - Progress: 92.04%
        - Current duration: 8.552s
        - ETA: 0.7396963156069951s
Layer-37 done.
        - Progress: 95.22%
        - Current duration: 9.67s
        - ETA: 0.48563335054299905s
Layer-38 done.
        - Progress: 97.30%
        - Current duration: 10.861s
        - ETA: 0.3014988859307728s
Layer-39 done.
        - Progress: 98.70%
        - Current duration: 11.825s
        - ETA: 0.15609997373361395s
Layer-40 done.
        - Progress: 99.49%
        - Current duration: 12.394s
        - ETA: 0.06293179101326096s
Layer-41 done.
        - Progress: 99.84%
        - Current duration: 12.716s
        - ETA: 0.020084561293895575s
Layer-42 done.
        - Progress: 99.96%
        - Current duration: 12.828s
        - ETA: 0.005462654539346801s
Layer-43 done.
        - Progress: 100.00%
        - Current duration: 12.914s
        - ETA: 0.0003233896564758254s
Done. Exporting solution...
root@19bb3cef6106:/subsetlatticebuilder# 
```
