import os


OPTIMAL_THR_FOR_CKPT_DICT = {
    os.path.join('20210529_thesis_indata_fixed_e1_n2_train_mass_ss', 'v2_time'): [0.52, 0.58, 0.54, 0.56, 0.52, 0.58, 0.52, 0.50, 0.56, 0.48, 0.52],

    os.path.join('20210529_thesis_indata_fixed_e2_n2_train_mass_ss', 'v2_time'): [0.52, 0.52, 0.56, 0.60, 0.52, 0.56, 0.52, 0.52, 0.56, 0.52, 0.48],

    os.path.join('20210529_thesis_indata_fixed_e1_n2_train_mass_kc', 'v2_time'): [0.54, 0.62, 0.58, 0.64, 0.66, 0.52, 0.50, 0.64, 0.62, 0.52, 0.58],

    os.path.join('20210529_thesis_indata_5cv_e1_n2_train_moda_ss', 'v2_time'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],

    os.path.join('20210529_thesis_indata_5cv_e1_n2_train_inta_ss', 'v2_time'): [0.50, 0.46, 0.42, 0.46, 0.48, 0.52, 0.52, 0.48, 0.54, 0.58, 0.54, 0.52, 0.48, 0.40, 0.44],

    os.path.join('20210529_thesis_indata_5cv_e1_n2_train_mass_ss', 'v2_time'): [0.56, 0.50, 0.54, 0.60, 0.50, 0.54, 0.56, 0.54, 0.48, 0.58, 0.54, 0.50, 0.52, 0.58, 0.46],

    os.path.join('20210529_thesis_indata_5cv_e2_n2_train_mass_ss', 'v2_time'): [0.48, 0.54, 0.50, 0.54, 0.60, 0.60, 0.54, 0.50, 0.52, 0.58, 0.56, 0.58, 0.54, 0.54, 0.56],

    os.path.join('20210529_thesis_indata_5cv_e1_n2_train_mass_kc', 'v2_time'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    
    os.path.join('20210605_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_sourcestd_to_e1_n2_train_mass_ss', 'v2_time'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210605_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_sourcestd_to_e2_n2_train_mass_ss', 'v2_time'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210605_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_sourcestd_to_e1_n2_train_inta_ss', 'v2_time'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],

    os.path.join('20210605_from_20210529_thesis_indata_5cv_e1_n2_train_inta_ss_desc_sourcestd_to_e1_n2_train_mass_ss', 'v2_time'): [0.50, 0.46, 0.42, 0.46, 0.48, 0.52, 0.52, 0.48, 0.54, 0.58, 0.54, 0.52, 0.48, 0.40, 0.44],
    os.path.join('20210605_from_20210529_thesis_indata_5cv_e1_n2_train_inta_ss_desc_sourcestd_to_e2_n2_train_mass_ss', 'v2_time'): [0.50, 0.46, 0.42, 0.46, 0.48, 0.52, 0.52, 0.48, 0.54, 0.58, 0.54, 0.52, 0.48, 0.40, 0.44],
    os.path.join('20210605_from_20210529_thesis_indata_5cv_e1_n2_train_inta_ss_desc_sourcestd_to_e1_n2_train_moda_ss', 'v2_time'): [0.50, 0.46, 0.42, 0.46, 0.48, 0.52, 0.52, 0.48, 0.54, 0.58, 0.54, 0.52, 0.48, 0.40, 0.44],
    
    os.path.join('20210605_from_20210529_thesis_indata_5cv_e1_n2_train_mass_ss_desc_sourcestd_to_e1_n2_train_moda_ss', 'v2_time'): [0.56, 0.50, 0.54, 0.60, 0.50, 0.54, 0.56, 0.54, 0.48, 0.58, 0.54, 0.50, 0.52, 0.58, 0.46],
    os.path.join('20210605_from_20210529_thesis_indata_5cv_e1_n2_train_mass_ss_desc_sourcestd_to_e1_n2_train_inta_ss', 'v2_time'): [0.56, 0.50, 0.54, 0.60, 0.50, 0.54, 0.56, 0.54, 0.48, 0.58, 0.54, 0.50, 0.52, 0.58, 0.46],
    os.path.join('20210605_from_20210529_thesis_indata_5cv_e1_n2_train_mass_ss_desc_sourcestd_to_e2_n2_train_mass_ss', 'v2_time'): [0.56, 0.50, 0.54, 0.60, 0.50, 0.54, 0.56, 0.54, 0.48, 0.58, 0.54, 0.50, 0.52, 0.58, 0.46],

    os.path.join('20210605_from_20210529_thesis_indata_5cv_e2_n2_train_mass_ss_desc_sourcestd_to_e1_n2_train_moda_ss', 'v2_time'): [0.48, 0.54, 0.50, 0.54, 0.60, 0.60, 0.54, 0.50, 0.52, 0.58, 0.56, 0.58, 0.54, 0.54, 0.56],
    os.path.join('20210605_from_20210529_thesis_indata_5cv_e2_n2_train_mass_ss_desc_sourcestd_to_e1_n2_train_inta_ss', 'v2_time'): [0.48, 0.54, 0.50, 0.54, 0.60, 0.60, 0.54, 0.50, 0.52, 0.58, 0.56, 0.58, 0.54, 0.54, 0.56],
    os.path.join('20210605_from_20210529_thesis_indata_5cv_e2_n2_train_mass_ss_desc_sourcestd_to_e1_n2_train_mass_ss', 'v2_time'): [0.48, 0.54, 0.50, 0.54, 0.60, 0.60, 0.54, 0.50, 0.52, 0.58, 0.56, 0.58, 0.54, 0.54, 0.56],
    
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_scale-0.5'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_scale-0.6'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_scale-0.7'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_scale-0.8'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_scale-0.9'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_scale-1.0'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_scale-1.1'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_scale-1.2'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_scale-1.3'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_scale-1.4'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_scale-1.5'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_invert-value'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_invert-time'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_filter-0-2'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_filter-2-4'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_filter-4-8'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_filter-8-11'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_filter-10-16'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_perturbation_to_e1_n2_train_moda_ss', 'v2_time_filter-16-30'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210621_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_pink_to_e1_n2_train_pink_nn', 'v2_time_scale-1.0'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210621_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_pink_to_e1_n2_train_pink_nn', 'v2_time_scale-1.5'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    os.path.join('20210621_from_20210529_thesis_indata_5cv_e1_n2_train_moda_ss_desc_pink_to_e1_n2_train_pink_nn', 'v2_time_scale-2.0'): [0.50, 0.54, 0.58, 0.54, 0.58, 0.50, 0.58, 0.52, 0.52, 0.56, 0.56, 0.56, 0.66, 0.54, 0.40],
    
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_scale-0.5'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_scale-0.6'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_scale-0.7'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_scale-0.8'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_scale-0.9'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_scale-1.0'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_scale-1.1'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_scale-1.2'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_scale-1.3'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_scale-1.4'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_scale-1.5'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_invert-value'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_invert-time'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_filter-0-2'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_filter-2-4'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_filter-4-8'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_filter-8-11'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_filter-10-16'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210620_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_perturbation_to_e1_n2_train_mass_kc', 'v2_time_filter-16-30'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210621_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_pink_to_e1_n2_train_pink_nn', 'v2_time_scale-1.0'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210621_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_pink_to_e1_n2_train_pink_nn', 'v2_time_scale-1.5'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],
    os.path.join('20210621_from_20210529_thesis_indata_5cv_e1_n2_train_mass_kc_desc_pink_to_e1_n2_train_pink_nn', 'v2_time_scale-2.0'): [0.56, 0.56, 0.58, 0.60, 0.56, 0.56, 0.58, 0.56, 0.56, 0.70, 0.58, 0.56, 0.56, 0.62, 0.64],

    os.path.join('20210621_thesis_whole_5cv_e1_n2_train_cap_ss', 'v2_time_subjectsize100.0'): [0.48, 0.48, 0.44, 0.50, 0.46],
    os.path.join('20210621_thesis_whole_5cv_e2_n2_train_cap_ss', 'v2_time_subjectsize100.0'): [0.48, 0.48, 0.50, 0.56, 0.48],
    os.path.join('20210621_thesis_whole_5cv_e3_n2_train_cap_ss', 'v2_time_subjectsize100.0'): [0.44, 0.48, 0.54, 0.46, 0.46],
    
    os.path.join('20210625_thesis_macro_subjects_5cv_e1_n2_train_cap_ss', 'v2_time_subjectsize075.0'): [0.44, 0.46, 0.48, 0.46, 0.46],
    os.path.join('20210625_thesis_macro_subjects_5cv_e1_n2_train_cap_ss', 'v2_time_subjectsize050.0'): [0.38, 0.48, 0.50, 0.48, 0.48],
    os.path.join('20210625_thesis_macro_subjects_5cv_e1_n2_train_cap_ss', 'v2_time_subjectsize025.0'): [0.54, 0.50, 0.42, 0.60, 0.48],
    os.path.join('20210625_thesis_macro_subjects_5cv_e1_n2_train_cap_ss', 'v2_time_subjectsize012.5'): [0.54, 0.52, 0.50, 0.52, 0.56],

    os.path.join('20210625_thesis_micro_signals_5cv_e1_n2_train_cap_ss', 'v2_time_signalsize080'): [0.44, 0.46, 0.50, 0.48, 0.52],
    os.path.join('20210625_thesis_micro_signals_5cv_e1_n2_train_cap_ss', 'v2_time_signalsize040'): [0.50, 0.46, 0.50, 0.48, 0.50],
    os.path.join('20210625_thesis_micro_signals_5cv_e1_n2_train_cap_ss', 'v2_time_signalsize020'): [0.38, 0.46, 0.44, 0.50, 0.50],
    os.path.join('20210625_thesis_micro_signals_5cv_e1_n2_train_cap_ss', 'v2_time_signalsize010'): [0.44, 0.46, 0.48, 0.58, 0.50],
    os.path.join('20210625_thesis_micro_signals_5cv_e1_n2_train_cap_ss', 'v2_time_signalsize005'): [0.54, 0.56, 0.58, 0.56, 0.52],
    
    os.path.join('20210703_from_20210529_thesis_indata_5cv_e1_n2_train_mass_ss_desc_finetune_to_e1_n2_train_moda_ss', 'v2_time_signalsize100.0'): [0.50, 0.52, 0.56, 0.46, 0.52, 0.44, 0.54, 0.54, 0.48, 0.56, 0.50, 0.52, 0.54, 0.52, 0.46],
    os.path.join('20210703_from_20210529_thesis_indata_5cv_e1_n2_train_mass_ss_desc_finetune_to_e1_n2_train_moda_ss', 'v2_time_signalsize70.0'): [0.42, 0.40, 0.38, 0.36, 0.46, 0.38, 0.42, 0.46, 0.46, 0.32, 0.38, 0.40, 0.30, 0.38, 0.36],
    os.path.join('20210703_from_20210529_thesis_indata_5cv_e1_n2_train_mass_ss_desc_finetune_to_e1_n2_train_moda_ss', 'v2_time_signalsize20.0'): [0.26, 0.38, 0.44, 0.30, 0.44, 0.46, 0.44, 0.26, 0.34, 0.42, 0.26, 0.34, 0.18, 0.16, 0.34],
    os.path.join('20210703_from_20210529_thesis_indata_5cv_e1_n2_train_mass_ss_desc_finetune_to_e1_n2_train_moda_ss', 'v2_time_signalsize40.0'): [0.18, 0.28, 0.22, 0.34, 0.34, 0.32, 0.36, 0.44, 0.40, 0.44, 0.30, 0.30, 0.28, 0.24, 0.40],
    os.path.join('20210703_from_20210529_thesis_indata_5cv_e1_n2_train_mass_ss_desc_finetune_to_e1_n2_train_moda_ss', 'v2_time_signalsize10.0'): [0.36, 0.34, 0.44, 0.34, 0.50, 0.44, 0.30, 0.36, 0.32, 0.32, 0.34, 0.54, 0.24, 0.24, 0.32],
    
    os.path.join('20210703_from_20210621_thesis_whole_5cv_e1_n2_train_cap_ss_desc_finetune_to_e1_n2_train_moda_ss', 'v2_time_subjectsize100.0_signalsize100.0'): [0.52, 0.50, 0.48, 0.52, 0.48, 0.48, 0.50, 0.50, 0.50, 0.46, 0.52, 0.48, 0.48, 0.48, 0.50],
    os.path.join('20210703_from_20210621_thesis_whole_5cv_e1_n2_train_cap_ss_desc_finetune_to_e1_n2_train_moda_ss', 'v2_time_subjectsize100.0_signalsize70.0'): [0.42, 0.38, 0.32, 0.40, 0.44, 0.38, 0.46, 0.44, 0.30, 0.40, 0.42, 0.42, 0.34, 0.36, 0.34],
    os.path.join('20210703_from_20210621_thesis_whole_5cv_e1_n2_train_cap_ss_desc_finetune_to_e1_n2_train_moda_ss', 'v2_time_subjectsize100.0_signalsize40.0'): [0.38, 0.38, 0.40, 0.36, 0.36, 0.28, 0.36, 0.44, 0.30, 0.34, 0.42, 0.30, 0.26, 0.24, 0.36],
    os.path.join('20210703_from_20210621_thesis_whole_5cv_e1_n2_train_cap_ss_desc_finetune_to_e1_n2_train_moda_ss', 'v2_time_subjectsize100.0_signalsize20.0'): [0.46, 0.38, 0.38, 0.30, 0.54, 0.40, 0.28, 0.42, 0.34, 0.30, 0.32, 0.40, 0.38, 0.40, 0.46],
    os.path.join('20210703_from_20210621_thesis_whole_5cv_e1_n2_train_cap_ss_desc_finetune_to_e1_n2_train_moda_ss', 'v2_time_subjectsize100.0_signalsize10.0'): [0.46, 0.38, 0.38, 0.32, 0.46, 0.36, 0.24, 0.30, 0.40, 0.40, 0.38, 0.34, 0.32, 0.38, 0.42],
    
    os.path.join('20210705_from_20210621_thesis_whole_5cv_e1_n2_train_cap_ss_desc_sourcestd_to_e1_n2_train_mass_ss', 'v2_time_subjectsize100.0'): [0.48, 0.48, 0.44, 0.50, 0.46, 0.48, 0.48, 0.44, 0.50, 0.46, 0.48, 0.48, 0.44, 0.50, 0.46],
    os.path.join('20210705_from_20210621_thesis_whole_5cv_e1_n2_train_cap_ss_desc_sourcestd_to_e2_n2_train_mass_ss', 'v2_time_subjectsize100.0'): [0.48, 0.48, 0.44, 0.50, 0.46, 0.48, 0.48, 0.44, 0.50, 0.46, 0.48, 0.48, 0.44, 0.50, 0.46],
    os.path.join('20210705_from_20210621_thesis_whole_5cv_e1_n2_train_cap_ss_desc_sourcestd_to_e1_n2_train_moda_ss', 'v2_time_subjectsize100.0'): [0.48, 0.48, 0.44, 0.50, 0.46, 0.48, 0.48, 0.44, 0.50, 0.46, 0.48, 0.48, 0.44, 0.50, 0.46],
    os.path.join('20210705_from_20210621_thesis_whole_5cv_e1_n2_train_cap_ss_desc_sourcestd_to_e1_n2_train_inta_ss', 'v2_time_subjectsize100.0'): [0.48, 0.48, 0.44, 0.50, 0.46, 0.48, 0.48, 0.44, 0.50, 0.46, 0.48, 0.48, 0.44, 0.50, 0.46],
    
    os.path.join('20210706_thesis_micro_signals_5cv_e1_n2_train_moda_ss', 'v2_time_signalsize100'): [0.54, 0.56, 0.56, 0.54, 0.52, 0.52, 0.52, 0.40, 0.50, 0.58, 0.50, 0.64, 0.56, 0.50, 0.48],
    os.path.join('20210706_thesis_micro_signals_5cv_e1_n2_train_moda_ss', 'v2_time_signalsize070'): [0.40, 0.44, 0.48, 0.46, 0.34, 0.42, 0.50, 0.36, 0.40, 0.56, 0.52, 0.44, 0.42, 0.42, 0.30],
    os.path.join('20210706_thesis_micro_signals_5cv_e1_n2_train_moda_ss', 'v2_time_signalsize040'): [0.36, 0.50, 0.50, 0.46, 0.40, 0.50, 0.50, 0.42, 0.46, 0.20, 0.36, 0.40, 0.50, 0.10, 0.48],
    os.path.join('20210706_thesis_micro_signals_5cv_e1_n2_train_moda_ss', 'v2_time_signalsize020'): [0.26, 0.50, 0.18, 0.52, 0.32, 0.32, 0.52, 0.40, 0.44, 0.42, 0.44, 0.48, 0.42, 0.20, 0.24],
    os.path.join('20210706_thesis_micro_signals_5cv_e1_n2_train_moda_ss', 'v2_time_signalsize010'): [0.38, 0.52, 0.22, 0.50, 0.26, 0.12, 0.46, 0.50, 0.46, 0.38, 0.36, 0.38, 0.32, 0.24, 0.34],
    
}



