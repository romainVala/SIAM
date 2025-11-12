import os.path
import sys
import torch, numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, isdir
sys.stdout = open(os.devnull, 'w')
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
sys.stdout = sys.__stdout__
from predNN.paths import get_model_path_and_fold, addprefixtofilenames, get_parent_path, gfile
import json, re
import torchio as tio

def get_nn_predictor(
        use_tta: bool = False,
        device: torch.device = torch.device('cuda'),
        verbose: bool = False
):
    os.environ['nnUNet_compile'] = 'F'
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=verbose,
        verbose_preprocessing=verbose
    )

    if device == torch.device('cpu'):
        torch.set_num_threads(os.cpu_count())
    return predictor

def convert_to_canonical_if_needed(file_list,voxel_size):
    fout_cano = addprefixtofilenames(file_list,'toCano_')
    fin, fout = [],[]
    tcano = tio.ToCanonical()
    reoriented_list = []
    for fi,fo in zip(file_list, fout_cano):
        ti = tio.ScalarImage(fi)
        if (ti.orientation == ('R', 'A', 'S') ) & (voxel_size==0) :
            fout.append(fi)
            reoriented_list.append(False)
        else:
            tc = tcano(ti)
            reoriented_list.append(True)
            if voxel_size>0:
                print(f'CHANING NIFTI voxel size to {voxel_size} saving {fo}')
                new_aff = tc.affine
                # alway >0 because tcano
                new_aff[0,0] = voxel_size
                new_aff[1,1] = voxel_size
                new_aff[2,2] = voxel_size
                tc.affine = new_aff #already load so all good
                #tc = tio.LabelMap(tensor=tc.data, affine=new_aff)
            else:
                print(f'cononical saving {fo}')

            tc.save(fo)
            fout.append(fo)

    return fout, reoriented_list

def nn_predict(
        input_file_or_folder: str,
        output_file_or_folder: str,
        use_tta: bool = False,
        device: torch.device = torch.device('cuda'),
        num_model: int =1,
        voxel_size=0,
        verbose: bool = False,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=8,
):

    folder_with_parameter_files, existing_fold, out_prefix = get_model_path_and_fold(num_model)

    # find input file or files
    if not input_file_or_folder.startswith('/'):
        input_file_or_folder = join(os.path.curdir, input_file_or_folder)
    input_is_file = True
    if os.path.isdir(input_file_or_folder):
        input_is_file = False
        input_file_orig =  gfile(input_file_or_folder,'.*nii')
        input_files, converted_to_canonical = convert_to_canonical_if_needed(input_file_orig,voxel_size)

        # output_file_or_folder must be folder in this case
        if output_file_or_folder is None:
            output_files = addprefixtofilenames(input_files, out_prefix)
            output_files_wanted = addprefixtofilenames(input_file_orig, out_prefix)

        else:
            if not output_file_or_folder.startswith('/'):
                output_file_or_folder = join( input_file_or_folder, f'{out_prefix}{output_file_or_folder}')
                print(f'setting output dir to {output_file_or_folder}')

            maybe_mkdir_p(output_file_or_folder)
            output_files = [join(output_file_or_folder, f'{out_prefix}{os.path.basename(i)}') for i in input_files]
            output_files_wanted = [join(output_file_or_folder, f'{out_prefix}{os.path.basename(i)}') for i in input_file_orig]


    else:
        #assert not isdir(output_file_or_folder), 'If input is a single file then output must be a filename, not a directory'
        #assert output_file_or_folder.endswith('.nii.gz'), 'Output file must end with .nii.gz'
        if output_file_or_folder is not None:
            out_prefix += output_file_or_folder

        input_files, converted_to_canonical = convert_to_canonical_if_needed([input_file_or_folder],voxel_size)
        input_file_orig =  [input_file_or_folder]
        output_files_wanted = addprefixtofilenames(input_file_orig,out_prefix)

        output_files = addprefixtofilenames(input_files,out_prefix)
        if os.path.isfile(output_files[0]):
            print(f'Existing output file {output_files} \n Skiping')
            return


    # Model prediction
    predictor = get_nn_predictor(
        use_tta=use_tta,
        device=device,
        verbose=verbose
    )

    # remove extension TODO only works if nii.gz
    fname = get_parent_path(input_files[0])[1]

    fname_ext = '.nii.gz' if fname.endswith('.nii.gz') else '.nii'
    fname_noext = fname[:-len(fname_ext)]
    #fname_noext = fname_noext if fname_noext.endswith('_0000') else fname_noext + '_0000'

    output_files = [oo[:-len(fname_ext)] for oo in output_files]

    if isinstance(folder_with_parameter_files,list):
        #cascade network or subregion
        diroutput = os.path.dirname(output_files[0])
        dirout1 = os.path.join(diroutput,'previous_stage')
        maybe_mkdir_p(dirout1)
        output_files_first_stage=[]
        for oo in output_files:
            file_basname = os.path.basename(oo)
            file_basname = file_basname[len(out_prefix):]
            #file_basname = file_basname[:-12] + '.nii.gz' no more extension
            file_basname = file_basname[:-5]
            output_files_first_stage.append( os.path.join(dirout1, file_basname) )
        print(f'first output file is {output_files_first_stage}')
        predictor.initialize_from_trained_model_folder(
            folder_with_parameter_files[0],
            existing_fold[0]
        )
        predictor.predict_from_files(
            [[i] for i in input_files],
            output_files_first_stage,
            save_probabilities=False,
            overwrite=False,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )
        if isinstance(folder_with_parameter_files[1],list): #subregion
            fjson = gfile(get_parent_path(output_files_first_stage)[0],'dataset.json')[0]
            with open(fjson) as f:
                dsj = json.load(f)
            dic_lab = dsj['labels']
            sub_model_name = get_parent_path(folder_with_parameter_files[1],2)[1]
            region=[]
            for DSname in sub_model_name:
                ind_underscore = [m.start() for m in re.finditer('_',DSname)]
                region.append(DSname[ind_underscore[0]+1:ind_underscore[1]])

            dirout_list = [os.path.join(dirout1,f'pred_{reg}') for reg in region]

            #make region prediction
            next_input_file = [[] for mm in region]
            for fin,fo in zip(input_files,output_files_first_stage):
                fname = get_parent_path(fin)[1]
                fin = os.path.abspath(fin)

                fname_ext = '.nii.gz' if fname.endswith('.nii.gz') else '.nii'
                fname_noext = fname[:-len(fname_ext)]
                fname_noext = fname_noext if fname_noext.endswith('_0000') else fname_noext+'_0000'
                fo = fo + fname_ext
                il = tio.LabelMap(fo)
                for ii, (reg, dirout2) in enumerate( zip(region, dirout_list) ):
                    data = il.data == dic_lab[reg]
                    iout = tio.LabelMap(tensor=data, affine=il.affine)
                    maybe_mkdir_p(dirout2)
                    fname1 = os.path.join(dirout2, fname_noext + fname_ext)
                    fname2 = os.path.join(dirout2, fname_noext[:-1] + '1' + fname_ext )
                    if not os.path.isfile(fname1):
                        os.symlink(fin, fname1)
                    iout.save(fname2)
                    next_input_file[ii].append([fname1[:-len(fname_ext)],fname2[:-len(fname_ext)]])

            output_files_regions=[] #[[] for mm in region]
            for ii, (mfolder, efold) in enumerate(zip(folder_with_parameter_files[1], existing_fold[1])) :
                #check existing oupt (since I delete the proba nnunet think it needs to be done
                output_files_reg, output_files_reg2 = [],[]
                for ffi in next_input_file[ii]:
                    fo = addprefixtofilenames(ffi[0],'pred_')[0]
                    output_files_reg2.append(fo) #need if pred is skip but not merge TODO remove
                    if os.path.isfile(fo+fname_ext) and ~ os.path.isfile(fo+'.npz'):
                        print(f'skiping {fo} because exist')
                    else:
                        output_files_reg.append(fo)
                output_files_regions.append(output_files_reg2)
                if len(output_files_reg)>0:
                    predictor.initialize_from_trained_model_folder(mfolder,efold )
                    predictor.predict_from_files(
                        next_input_file[ii],
                        output_files_reg,
                        save_probabilities=True,
                        overwrite=False,
                        num_processes_preprocessing=2,
                        num_processes_segmentation_export=1,
                        folder_with_segs_from_prev_stage=None,
                        num_parts=1,
                        part_id=0
                    )
                    #now force the prediction to be without BG into the initial mask
                    for iiff, fo in enumerate(output_files_reg):
                        if os.path.isfile(fo+fname_ext) and ~ os.path.isfile(fo+'.npz'):
                            print(f'skiping {fo} because exist')
                            continue

                        print(f'{ii} Merging : {fo+fname_ext}')
                        im = tio.LabelMap(next_input_file[ii][iiff][1]+fname_ext)
                        pred_logit = np.load(fo + ".npz")
                        data =  torch.from_numpy(pred_logit["probabilities"].transpose([0, 3,2,1]))
                        arr_shape = data.shape
                        data[0,: ] = torch.zeros(arr_shape[1:])  #force to predict anything but to 0
                        seg = data.argmax(0).unsqueeze(0) * im.data # mask with input mask
                        iout = tio.LabelMap(fo + fname_ext)
                        vol_change = (seg>0).sum() / (iout.data>0).sum()
                        print(f'extending predicted volume of {vol_change}') # value above 1.05 mean the model had hard time to predict (and prefered BG)
                        iout['data'] = seg
                        iout.save(fo + fname_ext)
                        os.remove(fo +".npz")
                        os.remove(fo + ".pkl")
            #merge all labels or todo one by one ?
            ind_label, ind_region = 0, 0
            dic_region_list = []
            for one_region in dic_lab.keys():
                if one_region in region:
                    #del dic_lab[one_region]
                    fjson = gfile(dirout_list[ind_region], 'dataset.json')[0]
                    with open(fjson) as f:
                        dic_subregion = json.load(f)['labels']
                    del dic_subregion['background']
                    for k in dic_subregion.keys():
                        dic_subregion[k] = ind_label
                        ind_label+=1
                    dic_region_list.append(dic_subregion)
                    ind_region+=1
                else:
                    dic_lab[one_region] = ind_label
                    ind_label += 1
            tmap_no_region = tio.RemapLabels({iii:vv for iii,vv in  enumerate(dic_lab.values())})
            for one_region in region:
                del dic_lab[one_region]
            for dd in dic_region_list:
                dic_lab.update(dd)
            for ii, (fi, fo) in enumerate( zip(output_files_first_stage, output_files) ):
                il = tmap_no_region(tio.LabelMap(fi+fname_ext))
                for fregion, dic_map in zip(output_files_regions, dic_region_list):
                    tmap = tio.RemapLabels({ind_val+1:v for ind_val,v in enumerate(dic_map.values())})
                    iregion = tmap( tio.LabelMap(fregion[ii]+fname_ext) )
                    mask = iregion.data>0
                    il['data'], iregion['data'] = il['data'].type(torch.ByteTensor), iregion['data'].type(torch.ByteTensor)
                    il['data'][mask] = iregion['data'][mask]
                il.save(fo+fname_ext)

            dic_lab = {k: v for k, v in sorted(dic_lab.items(), key=lambda item: item[1])}
            fo_dic = get_parent_path((fo))[0]+'/dataset_region_labels.json'
            with open(fo_dic, 'w') as file:
                json.dump(dic_lab, file, indent=4, sort_keys=False)



        else:         # second network cascade
            predictor.initialize_from_trained_model_folder(
                folder_with_parameter_files[1],
                existing_fold[1]
            )
            predictor.predict_from_files(
                [[i] for i in input_files],
                output_files,
                save_probabilities=False,
                overwrite=False,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=1,
                folder_with_segs_from_prev_stage=dirout1,
                num_parts=1,
                part_id=0
            )


    else:
        predictor.initialize_from_trained_model_folder(
            folder_with_parameter_files,
            existing_fold
        )

        predictor.predict_from_files(
            [[i] for i in input_files],
            output_files,
            save_probabilities=False,
            overwrite=False,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )

    for fi1,fi2, fo1, fo2, reorient_back in zip(input_file_orig,input_files,output_files_wanted,output_files, converted_to_canonical):
        if reorient_back:
            print('remove cana')
            tr = tio.Resample(target=fi1)
            if len(fname_ext)==4  : #ie input nifti .nii
                ffo = fo2 + fname_ext + '.gz'
                fo1 = fo1 + '.gz'  #force output to be in gz, important to spare disque especialy with label maps
            else:
                ffo =  fo2+fname_ext
            ipred = tio.LabelMap(ffo)
            ipred.load() #IMPORTANT to be able to modify the affine (the affine is not set, it is read "online")
            if voxel_size>0: #change resolution back
                ilorig = tio.ToCanonical()(tio.LabelMap(fi1)) #because we are in cano space
                # because we are in canonical with positif diag. we need to keep  >0 value so that the reslice reoder correctly
                ipred.affine[0, 0] = np.abs(ilorig.affine[0, 0])
                ipred.affine[1, 1] = np.abs(ilorig.affine[1, 1])
                ipred.affine[2, 2] = np.abs(ilorig.affine[2, 2])
            io = tr(ipred)

            io.save(fo1)
            print(f'removing {ffo}')
            print(f'removing {fi2}')
            os.remove(ffo)
            os.remove(fi2)



    if input_is_file:  #otherwise let it in the output folder
        try :
            # remove unnecessary json files
            # may fail if two single file prediction are launch in the same folder (because first process already removed the dataset.json)
            # not a big deal so skip it if errors
            label_file = join(os.path.dirname(output_files[0]), f'label_{out_prefix}.json')
            os.replace(join(os.path.dirname(output_files[0]), 'dataset.json'), label_file)
            #os.remove(join(os.path.dirname(output_files[0]), 'dataset.json'))
            os.remove(join(os.path.dirname(output_files[0]), 'plans.json'))
            os.remove(join(os.path.dirname(output_files[0]), 'predict_from_raw_data_args.json'))
        except:
            we_do_not_care=1

