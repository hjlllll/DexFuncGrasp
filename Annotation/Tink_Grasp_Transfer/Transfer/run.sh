python mesh_to_obj.py
python generate_sdf.py
python train_deep_sdf.py -e ../Dataset/Objects/bottle/split
python reconstruct_train.py -e ../Dataset/Objects/bottle/split --mesh_include
python tink/gen_interpolate.py --all -d ../Dataset/Objects/bottle/split

python tink/cal_contact_info_shadow.py -d ../Dataset/Objects/bottle/split -s bottle26 --tag trans
python tink/info_transform.py -d ../Dataset/Objects/bottle/split -s bottle26 -t bottle27 -p ../Dataset/Objects/bottle/split/contact/bottle26/trans
python tink/pose_refine.py -d ../Dataset/Objects/bottle/split -s bottle26 -t bottle12 -p ../Dataset/Objects/bottle/split/contact/bottle26/trans --vis