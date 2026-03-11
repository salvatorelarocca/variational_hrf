# python ~/variational_hrf/compute_wd.py \
#     --exp_name exp_vegeta \
#     --output_dir ./test \
#     --nfe_list 10,50,100,200 \
#     --num_samples 10 \
#     --num_projections 10 \
#     --batch_size 128 \
#     --gpu -1

# python ~/variational_hrf/compute_fid.py \
#     --exp_name exp_vegeta \
#     --output_dir ./test \
#     --N_list 2,2 \
#     --M_list 10,25 \
#     --num_samples 10 \
#     --gpu -1

# RF baseline — M_list non serve
python ~/variational_hrf/compute_fid.py \
    --exp_name exp_vegeta \
    --output_dir ./test \
    --N_list 10,50,100 \
    --num_samples 20 \
    --gpu -1 \
    --num_workers 0

