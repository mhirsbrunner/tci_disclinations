nx=16
nz=16
half_model=False
other_half=False
spin=-1

mkdir data
for mass in -4. -3.9 -3.8 -3.7 -3.6 -3.5 -3.4 -3.3 -3.2 -3.1 -2.9 -2.8 -2.7 -2.6 -2.5 -2.4 -2.3 -2.2 -2.1 -2. -1.9 -1.8 -1.7 -1.6 -1.5 -1.4 -1.3 -1.2 -1.1 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1  0. 0.1  0.2  0.3 0.4  0.5  0.6  0.7  0.8  0.9  1.1  1.2  1.3  1.4 1.5  1.6  1.7  1.8  1.9  2.  2.1  2.2  2.3  2.4  2.5 2.6  2.7  2.8  2.9  3.1  3.2  3.3  3.4  3.5  3.6 3.7 3.8  3.9 4.0
#for mass in 2.0
do
    echo $mass
    sbatch --job-name="mass_"$mass \
    --export=ALL,nx=$nx,nz=$nz,mass=$mass,half_model=$half_model,other_half=$other_half,spin=$spin job.slurm
done
