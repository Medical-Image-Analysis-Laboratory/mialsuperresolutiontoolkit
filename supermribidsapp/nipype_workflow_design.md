# Super MRI BIDS App - BIDS/Nipype workflow

 1. Execution template

     ``` docker run -u $(id -u):$(id -g) -v /local/dir/for/bids/dataset:/tmp --rm -it sebastientourbier(brainhack_ch)/supermribidsapp:latest /tmp /tmp/derivatives participant --participant_label eyesVD46a --pipeline_config /tmp/supermri_config.txt```

     ```-u: set appropriate username and group```

     ```-v: map local directory to directory within the docker image```

 2. Config template for `/tmp/supermri_config.txt`
 ```
 [Preprocess]
 nlm_weight=0.1

 [Reconstruction]
 deltat=0.01
 lambda=0.75
 ```

 3. load config file

   ```python
   def anat_load_config(pipeline, config_path):
      config = ConfigParser.ConfigParser()
      config.read(config_path)
      global_keys = [prop for prop in pipeline.global_conf.traits().keys() if not 'trait' in prop] # possibly dangerous..?
    for key in global_keys:
        if key != "subject" and key != "subjects" and key != "subject_session" and key != "subject_sessions" and key != 'modalities':
            conf_value = config.get('Global', key)
            setattr(pipeline.global_conf, key, conf_value)
    for stage in pipeline.stages.values():
        stage_keys = [prop for prop in stage.config.traits().keys() if not 'trait' in prop] # possibly dangerous..?
        for key in stage_keys:
            if 'config' in key: #subconfig
                sub_config = getattr(stage.config, key)
                stage_sub_keys = [prop for prop in sub_config.traits().keys() if not 'trait' in prop]
                for sub_key in stage_sub_keys:
                    try:
                        conf_value = config.get(stage.name, key+'.'+sub_key)
                        try:
                            conf_value = eval(conf_value)
                        except:
                            pass
                        setattr(sub_config, sub_key, conf_value)
                    except:
                        pass
            else:
                try:
                    conf_value = config.get(stage.name, key)
                    try:
                        conf_value = eval(conf_value)
                    except:
                        pass
                    setattr(stage.config, key, conf_value)
                except:
                    pass
    setattr(pipeline,'number_of_cores',int(config.get('Multi-processing','number_of_cores')))

    return True
    ```

 4. Workflow

 mialsrtkOrientImage -> btkNLMDenoising ->  mialsrtkCorrectSliceIntensity -> mialsrtkSliceBySliceN4BiasFieldCorrection -> mialsrtkSliceBySliceCorrectBiasField -> mialsrtkCorrectSliceIntensity -> mialsrtkHistogramNormalization -> mialsrtkIntensityStandardization -> mialsrtkImageReconstruction -> mialsrtkTVSuperResolution -> mialsrtkRefineHRMaskByIntersection -> mialsrtkN4BiasFieldCorrection
