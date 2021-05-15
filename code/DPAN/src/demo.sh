gpu_id=$1


python main.py --gpu_id ${gpu_id} --template DPAN --scale 2 --patch_size 96 --save DPAN_G10B10_DF_X2 

# validation
python main.py --template DPAN --data_test Set5+Set14 --scale 2 --pre_train ../experiment/DPAN_G10B10_X2/model/model_best.pt  --test_only --chop 
#--save_results 
#--self_ensemble


