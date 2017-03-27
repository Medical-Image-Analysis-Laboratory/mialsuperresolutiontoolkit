# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/tourbier/Desktop/FetalBrainToolkit3/fetalbrainextraction.ui'
#
# Created: Mon Sep 23 15:24:44 2013
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

import glob,os,shlex,subprocess,sys,time
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QApplication
#from nipype.interfaces.slicer import BRAINSFit, BRAINSResample
#import nipype.pipeline.engine as pe 

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(510, 314)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.DataHorizontalLayout = QtGui.QHBoxLayout()
        self.DataHorizontalLayout.setObjectName(_fromUtf8("DataHorizontalLayout"))
        self.data_label = QtGui.QLabel(self.centralwidget)
        self.data_label.setObjectName(_fromUtf8("data_label"))
        self.DataHorizontalLayout.addWidget(self.data_label)
        self.data_lineEdit = QtGui.QLineEdit(self.centralwidget)
        self.data_lineEdit.setObjectName(_fromUtf8("data_lineEdit"))
        self.DataHorizontalLayout.addWidget(self.data_lineEdit)
        self.browse_button = QtGui.QToolButton(self.centralwidget)
        self.browse_button.setObjectName(_fromUtf8("browse_button"))
        self.DataHorizontalLayout.addWidget(self.browse_button)
        self.setDir_button = QtGui.QPushButton(self.centralwidget)
        self.setDir_button.setObjectName(_fromUtf8("setDir_button"))
        self.DataHorizontalLayout.addWidget(self.setDir_button)
        self.gridLayout.addLayout(self.DataHorizontalLayout, 0, 0, 1, 1)
        self.PatientHorizontalLayout_4 = QtGui.QHBoxLayout()
        self.PatientHorizontalLayout_4.setObjectName(_fromUtf8("PatientHorizontalLayout_4"))
        self.patient_label = QtGui.QLabel(self.centralwidget)
        self.patient_label.setObjectName(_fromUtf8("patient_label"))
        self.PatientHorizontalLayout_4.addWidget(self.patient_label)
        self.patient_lineEdit = QtGui.QLineEdit(self.centralwidget)
        self.patient_lineEdit.setObjectName(_fromUtf8("patient_lineEdit"))
        self.PatientHorizontalLayout_4.addWidget(self.patient_lineEdit)
        self.gridLayout.addLayout(self.PatientHorizontalLayout_4, 1, 0, 1, 1)
        self.AtlasHorizontalLayout_2 = QtGui.QHBoxLayout()
        self.AtlasHorizontalLayout_2.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.AtlasHorizontalLayout_2.setObjectName(_fromUtf8("AtlasHorizontalLayout_2"))
        self.atlas_label = QtGui.QLabel(self.centralwidget)
        self.atlas_label.setObjectName(_fromUtf8("atlas_label"))
        self.AtlasHorizontalLayout_2.addWidget(self.atlas_label)
        self.atlas_listWidget = QtGui.QListWidget(self.centralwidget)
        self.atlas_listWidget.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        self.atlas_listWidget.setLayoutMode(QtGui.QListView.SinglePass)
        self.atlas_listWidget.setObjectName(_fromUtf8("atlas_listWidget"))
        self.AtlasHorizontalLayout_2.addWidget(self.atlas_listWidget)
        self.gridLayout.addLayout(self.AtlasHorizontalLayout_2, 2, 0, 1, 1)

        self.method_groupBox = QtGui.QGroupBox(self.centralwidget)
        self.method_groupBox.setObjectName(_fromUtf8("groupBox"))
        self.methodHorizontalLayout = QtGui.QHBoxLayout(self.method_groupBox)
        self.methodHorizontalLayout.setObjectName(_fromUtf8("methodHorizontalLayout"))
        self.mv_radioButton = QtGui.QRadioButton(self.method_groupBox)
        self.mv_radioButton.setObjectName(_fromUtf8("mv_radioButton"))
        self.mv_radioButton.setChecked(True)
        self.buttonGroup = QtGui.QButtonGroup(self.centralwidget)
        self.buttonGroup.setObjectName(_fromUtf8("buttonGroup"))
        self.buttonGroup.addButton(self.mv_radioButton)
        self.methodHorizontalLayout.addWidget(self.mv_radioButton)
        self.gwv_radioButton = QtGui.QRadioButton(self.method_groupBox)
        self.gwv_radioButton.setObjectName(_fromUtf8("gwv_radioButton"))
        self.buttonGroup.addButton(self.gwv_radioButton)
        self.methodHorizontalLayout.addWidget(self.gwv_radioButton)
        self.lwv_radioButton = QtGui.QRadioButton(self.method_groupBox)
        self.lwv_radioButton.setObjectName(_fromUtf8("lwv_radioButton"))
        self.buttonGroup.addButton(self.lwv_radioButton)
        self.methodHorizontalLayout.addWidget(self.lwv_radioButton)

        self.gridLayout.addWidget(self.method_groupBox, 3, 0, 1, 1)

        self.radiusHorizontalLayout = QtGui.QHBoxLayout()
        self.radiusHorizontalLayout.setMargin(0)
        self.radiusHorizontalLayout.setObjectName(_fromUtf8("radiusHorizontalLayout"))
        self.radius_label = QtGui.QLabel(self.centralwidget)
        self.radius_label.setObjectName(_fromUtf8("radius_label"))
        self.radiusHorizontalLayout.addWidget(self.radius_label)
        self.radius_spinBox = QtGui.QSpinBox(self.centralwidget)
        self.radius_spinBox.setProperty("value", 1)
        self.radius_spinBox.setObjectName(_fromUtf8("radius_spinBox"))
        self.radius_spinBox.setRange(1,10)
        self.radiusHorizontalLayout.addWidget(self.radius_spinBox)

        self.radius_label.setDisabled(True)
        self.radius_spinBox.setDisabled(True)

        self.gridLayout.addLayout(self.radiusHorizontalLayout, 4, 0, 1, 1)

        self.reg_radioButton = QtGui.QRadioButton(self.centralwidget)
        self.reg_radioButton.setObjectName(_fromUtf8("reg_radioButton"))
        self.reg_radioButton.setChecked(True)

        self.eval_radioButton = QtGui.QRadioButton(self.centralwidget)
        self.eval_radioButton.setObjectName(_fromUtf8("reg_radioButton"))
        self.eval_radioButton.setChecked(True)

        self.RunHorizontalLayout_3 = QtGui.QHBoxLayout()
        self.RunHorizontalLayout_3.setObjectName(_fromUtf8("RunHorizontalLayout_3"))
        self.logo_label = QtGui.QLabel(self.centralwidget)
        self.logo_label.setText(_fromUtf8(""))
        self.logo_label.setPixmap(QtGui.QPixmap(_fromUtf8(MainWindow.executable_dir+"/Resources/MIALlogo220x60.png")))
        self.logo_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.logo_label.setObjectName(_fromUtf8("logo_label"))
        self.RunHorizontalLayout_3.addWidget(self.logo_label)
        self.extract_button = QtGui.QPushButton(self.centralwidget)
        self.extract_button.setObjectName(_fromUtf8("extract_button"))
        self.RunHorizontalLayout_3.addWidget(self.extract_button)
        self.gridLayout.addLayout(self.RunHorizontalLayout_3, 5, 0, 1, 1)
        self.statusBar = QtGui.QStatusBar(self.centralwidget)
        #self.progressBar = QtGui.QProgressBar(self.statusBar)
        self.gridLayout.addWidget(self.statusBar)
        self.statusBar.showMessage("Extraction Status | Setup")
        MainWindow.setCentralWidget(self.centralwidget)

        QtCore.QObject.connect(self.extract_button, QtCore.SIGNAL('clicked()'), MainWindow.extractOnClicked)
        QtCore.QObject.connect(self.browse_button, QtCore.SIGNAL('clicked()'), MainWindow.browseOnClicked)
        QtCore.QObject.connect(self.setDir_button, QtCore.SIGNAL('clicked()'), MainWindow.setDirOnClicked)
        QtCore.QObject.connect(self.buttonGroup, QtCore.SIGNAL('buttonClicked(int)'), MainWindow.setFusionOnClicked)
        QtCore.QObject.connect(self.radius_spinBox, QtCore.SIGNAL('valueChanged(int)'), MainWindow.setPatchRadiusOnValueChanged)


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "Fetal brain extraction", None, QtGui.QApplication.UnicodeUTF8))
        self.data_label.setText(QtGui.QApplication.translate("MainWindow", "Data folder :", None, QtGui.QApplication.UnicodeUTF8))
        self.browse_button.setText(QtGui.QApplication.translate("MainWindow", "...", None, QtGui.QApplication.UnicodeUTF8))
        self.setDir_button.setText(QtGui.QApplication.translate("MainWindow", "Set", None, QtGui.QApplication.UnicodeUTF8))
        self.patient_label.setText(QtGui.QApplication.translate("MainWindow", "Patient :", None, QtGui.QApplication.UnicodeUTF8))
        self.atlas_label.setText(QtGui.QApplication.translate("MainWindow", "Atlases :", None, QtGui.QApplication.UnicodeUTF8))
        self.method_groupBox.setTitle(QtGui.QApplication.translate("MainWindow", "Fusion :", None, QtGui.QApplication.UnicodeUTF8))
        self.mv_radioButton.setText(QtGui.QApplication.translate("MainWindow", "Majority voting", None, QtGui.QApplication.UnicodeUTF8))
        self.gwv_radioButton.setText(QtGui.QApplication.translate("MainWindow", "Global weighted voting", None, QtGui.QApplication.UnicodeUTF8))
        self.lwv_radioButton.setText(QtGui.QApplication.translate("MainWindow", "Local weighted voting", None, QtGui.QApplication.UnicodeUTF8))
        self.radius_label.setText(QtGui.QApplication.translate("MainWindow", "Patch radius (LWV)", None, QtGui.QApplication.UnicodeUTF8))
        #self.method_label.setText(QtGui.QApplication.translate("MainWindow", "Fusion :", None, QtGui.QApplication.UnicodeUTF8))
        self.radius_label.setText(QtGui.QApplication.translate("MainWindow", "Patch radius (LWV) :", None, QtGui.QApplication.UnicodeUTF8))
        self.atlas_listWidget.setSortingEnabled(True)
        self.extract_button.setText(QtGui.QApplication.translate("MainWindow", "Extract Brain", None, QtGui.QApplication.UnicodeUTF8))


class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent = None):
        self.executable_dir="/".join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
	print self.executable_dir
        self.brainstools_dir=os.getenv('BRAINSTOOLS_DIR')
        self.mialsrtk_utilities_dir= os.getenv('MIALSRTK_DIR') + '/Utilities'
        self.data_directory=""
        self.patient=""

        QtGui.QMainWindow.__init__(self,parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setupAtlasList()

        self.fusion_method_id = '0'
        self.patch_radius = self.ui.radius_spinBox.value()
        

    def setupAtlasList(self):
        atlases = [
            'F45248',
            'F570362',
            'F1094830',
            'F2234690',
            'F2823438',
            'F2850448',
            'F2881456',
            'F2884416',
            'Patient01',
            'Patient03'
        ]
        for atlas in atlases:
            item = QtGui.QListWidgetItem(atlas)
            self.ui.atlas_listWidget.addItem(item)


    def extractOnClicked(self):
        print 'button extract clicked'

        run_evaluation = True;

        self.selected_atlases = [str(atlas.text()) for atlas in self.ui.atlas_listWidget.selectedItems()]
        
        if not self.data_directory:
            print 'Error 1: Data folder not set!'
        elif not str(self.ui.patient_lineEdit.text()):
            print 'Error 2: Patient name not set!'
        elif len(self.selected_atlases) == 0:
            print 'Error 3: No atlas selected!'
        else:
            tic = time.clock()
            print 'Start brain extraction...'
            print 'Atlas selection:' 
            print self.selected_atlases

            self.selected_atlases_dir=[]
            self.patient_name=str(self.ui.patient_lineEdit.text())

            self.patient_lr_dir=self.data_directory+'/'+self.patient_name+'/LR'
            self.patient_lr_paths=glob.glob(self.patient_lr_dir+'/*.nii')

            print 'Patient input data (LR stacks): ',self.patient_lr_dir

            self.patient_mask_dir=self.data_directory+'/'+self.patient_name+'/Masks'

            print 'Patient output data (Masks): ',self.patient_mask_dir

            self.ui.statusBar.showMessage("Extraction Status | Atlas brain segmentation (0%)")

            step=100/len(self.selected_atlases)

            for atlas_index in range(0,len(self.selected_atlases)):
                self.ui.statusBar.showMessage("Extraction Status | Atlas brain segmentation ("+str((atlas_index)*step)+"%) | Process Atlas #"+str(atlas_index+1)+" ("+self.selected_atlases[atlas_index]+")")
                #otsu_thresh_command = self.create_otsu_command(self.data_directory+'/'+self.selected_atlases[atlas_index]+'/SDI')
                #print otsu_thresh_command
                #otsu_return_code = subprocess.call(otsu_thresh_command)

            self.ui.statusBar.showMessage("Extraction Status | Atlas brain segmentation (100%)")               
            print len(self.patient_lr_paths), len(self.selected_atlases)
            step=100/(len(self.patient_lr_paths)*len(self.selected_atlases))    

            progress=0

            for atlas_index in range(0,len(self.selected_atlases)):
                self.selected_atlases_dir.append(self.data_directory+'/'+self.selected_atlases[atlas_index]+'/SDI')

            for lr_index in range(0,len(self.patient_lr_paths)):
                print 'Process image ',".".join(self.patient_lr_paths[lr_index].split('/')[-1].split('.')[:-1])
                for atlas_index in range(0,len(self.selected_atlases)):
                    self.ui.statusBar.showMessage("Extraction Status | Brain extraction ("+str(progress)+"%) | Process image "+".".join(self.patient_lr_paths[lr_index].split('/')[-1].split('.')[:-1])+" with Atlas "+self.selected_atlases[atlas_index])

                    print 'using atlas',self.selected_atlases[atlas_index]

                    #brainsfits_command = self.create_brainsfit_command(lr_index,self.patient_lr_paths[lr_index],self.selected_atlases_dir[atlas_index])
                    #print brainsfits_command
                    #brainsfit_return_code = subprocess.call(brainsfits_command)

                    #brainsresample_command = self.create_brainsresample_command(lr_index,self.patient_lr_paths[lr_index],self.patient_mask_dir,self.selected_atlases_dir[atlas_index],self.selected_atlases[atlas_index])
                    #print brainsresample_command
                    #brainsresample_return_code = subprocess.call(brainsresample_command)

                    if run_evaluation:
                        evaluation_command = self.create_evaluation_command(self.patient_name,self.selected_atlases[atlas_index],self.patient_mask_dir,self.patient_lr_paths[lr_index],'mono',str(self.ui.radius_spinBox.value()))
                        #print evaluation_command
                        #evaluation_return_code = subprocess.call(evaluation_command)

                    progress+=step

                #atlas_nuc_fusion_command = self.create_atlas_fusion_command(self.patient_name,self.patient_lr_paths[lr_index],self.patient_mask_dir,self.selected_atlases_dir,self.fusion_method_id,str(self.ui.radius_spinBox.value()))
                #print atlas_nuc_fusion_command
                #atlas_nuc_fusion_return_code = subprocess.call(atlas_nuc_fusion_command)

                atlas_nuc_fusion_old_ncc_command = self.create_atlas_nuc_fusion_old_ncc_command(self.patient_name,self.patient_lr_paths[lr_index],self.patient_mask_dir,self.selected_atlases_dir,self.fusion_method_id,str(self.ui.radius_spinBox.value()))
                print atlas_nuc_fusion_old_ncc_command
                atlas_nuc_fusion_old_ncc_return_code = subprocess.call(atlas_nuc_fusion_old_ncc_command)

                #atlas_fusion_command = self.create_atlas_nuc_fusion_command(self.patient_name,self.patient_lr_paths[lr_index],self.patient_mask_dir,self.selected_atlases_dir,self.fusion_method_id,str(self.ui.radius_spinBox.value()))
                #print atlas_fusion_command
                #atlas_fusion_return_code = subprocess.call(atlas_fusion_command)

                if run_evaluation:
                    #evaluation_command = self.create_evaluation_command(self.patient_name,self.selected_atlases[atlas_index],self.patient_mask_dir,self.patient_lr_paths[lr_index],'fusion',str(self.ui.radius_spinBox.value()))
                    #print evaluation_command
                    #evaluation_return_code = subprocess.call(evaluation_command)
                    #nuc_evaluation_command = self.create_nuc_evaluation_command(self.patient_name,self.selected_atlases[atlas_index],self.patient_mask_dir,self.patient_lr_paths[lr_index],'fusion',str(self.ui.radius_spinBox.value()))
                    #print nuc_evaluation_command
                    #nuc_evaluation_return_code = subprocess.call(nuc_evaluation_command)
                    nuc_evaluation_old_ncc_command = self.create_nuc_old_ncc_evaluation_command(self.patient_name,self.selected_atlases[atlas_index],self.patient_mask_dir,self.patient_lr_paths[lr_index],'fusion',str(self.ui.radius_spinBox.value()))
                    print nuc_evaluation_old_ncc_command
                    nuc_evaluation_old_ncc_return_code = subprocess.call(nuc_evaluation_old_ncc_command)
                 
                    #return_code = subprocess.call(brainsfits_command,shell=True)
            toc = time.clock()
            elapsed_time = toc -tic
            message = "Extraction Status | Done (elapsed time = ",elapsed_time,"s.)"
            self.ui.statusBar.showMessage(str(message))

    def create_otsu_command(self,atlas_dir):
        command=[]
        command_line='"'+self.mialsrtk_utilities_dir+'/mialsrtkOtsuThresholdSegmentation'+'" '
        command_line+='--brightObjects '
        command_line+='--numberOfBins "128" '
        command_line+='--faceConnected '
        command_line+='--minimumObjectSize "0" '
        command_line+='--input '+'"'+atlas_dir+'/SDI_NLM_BCORR_NORM.nii'+'" '
        command_line+='--output '+'"'+atlas_dir+'/SDI_NLM_otsu_mask.nii'+'" '
        print "Command line: ",command_line
        args = shlex.split(command_line)
        return args
        # command.append(self.mialsrtk_utilities_dir+'/stkOtsuThreshold/Bin/stkOtsuThresholdSegmentation')
        # command.append("--brightObjects")
        # command.append("--numberOfBins")
        # command.append("128")
        # command.append("--faceConnected")
        # command.append("--minimumObjectSize")
        # command.append("0")
        # command.append("--input")
        # command.append(atlas_dir+'/SDI_NLM_BCORR_NORM.nii')
        # command.append("--output")
        # command.append('"'+atlas_dir+'/SDI_NLM_otsu_mask.nii'+'"')
        #return command

    def create_brainsresample_command(self, lr_index, patient_lr_path, patient_mask_dir, atlas_dir, atlas_name):
        command=[]
        command.append(self.brainstools_dir+'/BRAINSResample')
        command.append('--inputVolume')
        command.append(atlas_dir+'/SDI_NLM_otsu_mask.nii')
        command.append('--referenceVolume')
        command.append(patient_lr_path)
        command.append('--outputVolume')
        command.append(patient_mask_dir+'/AutoExtracted/'+atlas_name+'/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_lin_mask.nii')
        command.append('--pixelType')
        command.append('float')
        command.append('--warpTransform')
        command.append(atlas_dir+'/Transformed/'+self.patient_name+'/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_transform.h5')
        command.append('--interpolationMode')
        command.append('NearestNeighbor')
        command.append('--defaultValue')
        command.append('0')
        command.append('--numberOfThreads')
        command.append('8')
        return command

    def create_brainsfit_command(self, lr_index, patient_lr_path, atlas_path):
        command=[]
        command.append(self.brainstools_dir+'/BRAINSFit')
        command.append('--costMetric')
        command.append('MMI')
        command.append('--fixedVolume')
        command.append(patient_lr_path)
        command.append('--movingVolume')
        command.append(atlas_path+'/SDI_NLM_BCORR_NORM.nii')
        command.append('--numberOfSamples')
        command.append('100000')
        command.append('--numberOfIterations')
        command.append('1500,1500,1500')
        command.append('--numberOfHistogramBins')
        command.append('50')
        command.append('--maximumStepLength')
        command.append('0.2')
        command.append('--minimumStepLength')
        command.append('0.005,0.005,0.005')
        command.append('--transformType')
        command.append('Rigid,ScaleVersor3D,BSpline')
        command.append('--relaxationFactor')
        command.append('0.5')
        command.append('--translationScale')
        command.append('1000')
        command.append('--reproportionScale')
        command.append('1')
        command.append('--skewScale')
        command.append('1')
        command.append('--useExplicitPDFDerivativesMode')
        command.append('AUTO')
        command.append('--useCachingOfBSplineWeightsMode')
        command.append('ON')
        command.append('--maxBSplineDisplacement')
        command.append('1')
        command.append('--projectedGradientTolerance')
        command.append('0')
        command.append('--costFunctionConvergenceFactor')
        command.append('1e+09')
        command.append('--backgroundFillValue')
        command.append('0')
        command.append('--initializeTransformMode')
        command.append('useMomentsAlign')
        command.append('--maskInferiorCutOffFromCenter')
        command.append('1000')
        command.append('--splineGridSize')
        command.append('10,14,12')
        command.append('--outputVolume')
        command.append(atlas_path+'/Transformed/'+self.patient_name+'/SDI_NLM_BCORR_NORM_'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_t.nii')
        command.append('--outputTransform')
        command.append(atlas_path+'/Transformed/'+self.patient_name+'/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_transform.h5')
        command.append('--numberOfThreads')
        command.append('8')
        return command

    def create_atlas_fusion_command(self,patient_name,patient_lr_path,patient_mask_dir,atlas_paths,fusion_method_id,patch_radius):
        command=[]
        command.append(self.mialsrtk_utilities_dir+'/mialsrtkSegmentationWeightedFusion');
        command.append('--output');

        if fusion_method_id == '0':
            command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_MV_mask.nii');
        elif fusion_method_id == '1':
            command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_GWV_mask.nii');
        elif fusion_method_id == '2':
            command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_LWV_'+patch_radius+'_mask.nii');

        for atlas_index in range(0,len(atlas_paths)):
            atlas_name = atlas_paths[atlas_index].split('/')[-2]
            command.append('-i');
            command.append(patient_lr_path);
            command.append('-r');
            command.append(atlas_paths[atlas_index]+'/Transformed/'+patient_name+'/SDI_NLM_BCORR_NORM_'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_t.nii');
            command.append('-m');
            command.append(patient_mask_dir+'/AutoExtracted/'+atlas_name+'/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_lin_mask.nii');

        if fusion_method_id == '1':
            command.append('--fusion-method');
            command.append('1');
        if fusion_method_id == '2':
            command.append('--fusion-method');
            command.append('2');
            command.append('--patch-radius');
            command.append(patch_radius);

        command.append('--profiling-csv');
        command.append(patient_mask_dir+'/timing_performance.csv');

        return command;

    def create_atlas_nuc_fusion_command(self,patient_name,patient_lr_path,patient_mask_dir,atlas_paths,fusion_method_id,patch_radius):
        command=[]
        command.append(self.mialsrtk_utilities_dir+'/mialsrtkSegmentationWeightedFusionWithUnanimousConsensus');
        command.append('--output');

        if fusion_method_id == '0':
            command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_NUCMV_mask.nii');
        elif fusion_method_id == '1':
            command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_NUCGWV_mask.nii');
        elif fusion_method_id == '2':
            command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_NUCLWV_'+patch_radius+'_mask.nii');

        command.append('--non-unanimous-consensus');
        command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_nuc_mask.nii');

        command.append('--unanimous-consensus');
        command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_ouc_mask.nii');

        for atlas_index in range(0,len(atlas_paths)):
            atlas_name = atlas_paths[atlas_index].split('/')[-2]
            command.append('-i');
            command.append(patient_lr_path);
            command.append('-r');
            command.append(atlas_paths[atlas_index]+'/Transformed/'+patient_name+'/SDI_NLM_BCORR_NORM_'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_t.nii');
            command.append('-m');
            command.append(patient_mask_dir+'/AutoExtracted/'+atlas_name+'/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_lin_mask.nii');

        if fusion_method_id == '1':
            command.append('--fusion-method');
            command.append('1');
        if fusion_method_id == '2':
            command.append('--fusion-method');
            command.append('2');
            command.append('--patch-radius');
            command.append(patch_radius);

        command.append('--profiling-csv');
        command.append(patient_mask_dir+'/timing_performance.csv');

        return command;

    def create_atlas_nuc_fusion_old_ncc_command(self,patient_name,patient_lr_path,patient_mask_dir,atlas_paths,fusion_method_id,patch_radius):
        command=[]
        command.append(self.mialsrtk_utilities_dir+'/mialsrtkSegmentationWeightedFusionWithUnanimousConsensus');
        command.append('--output');

        if fusion_method_id == '0':
            command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_NUCMV_itkNCC_mask.nii');
        elif fusion_method_id == '1':
            command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_NUCGWV_itkNCC_mask.nii');
        elif fusion_method_id == '2':
            command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_NUCLWV_itkNCC_'+patch_radius+'_mask.nii');

        command.append('--non-unanimous-consensus');
        command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_nuc_itkNCC_mask.nii');

        command.append('--unanimous-consensus');
        command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_ouc_itkNCC_mask.nii');

        for atlas_index in range(0,len(atlas_paths)):
            atlas_name = atlas_paths[atlas_index].split('/')[-2]
            command.append('-i');
            command.append(patient_lr_path);
            command.append('-r');
            command.append(atlas_paths[atlas_index]+'/Transformed/'+patient_name+'/SDI_NLM_BCORR_NORM_'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_t.nii');
            command.append('-m');
            command.append(patient_mask_dir+'/AutoExtracted/'+atlas_name+'/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_lin_mask.nii');

        if fusion_method_id == '1':
            command.append('--fusion-method');
            command.append('1');
        if fusion_method_id == '2':
            command.append('--fusion-method');
            command.append('2');
            command.append('--patch-radius');
            command.append(patch_radius);

        command.append('--profiling-csv');
        command.append(patient_mask_dir+'/timing_performance.csv');

        command.append('--use-itk-ncc');

        return command;

    def create_evaluation_command(self,patient_name,atlas_name,patient_mask_dir,patient_lr_path,segmentation_type,patch_radius):
        stack_name = patient_lr_path.split('/')[-1]
        prefix_stack_name = stack_name.split('_')[0]
        postfix_stack_name = stack_name.split('_')[1]

        command=[]
        command.append(self.mialsrtk_utilities_dir+'/mialsrtkEvaluateLabelOverlapMeasures');
        command.append('--ref');
        command.append(patient_mask_dir+'/'+prefix_stack_name+'_mask_'+postfix_stack_name);

        if segmentation_type == 'mono':
            command.append('--input');
            command.append(patient_mask_dir+'/AutoExtracted/'+atlas_name+'/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_lin_mask.nii');
        elif segmentation_type == 'fusion':
            command.append('--input');
            if self.fusion_method_id == '0':
                command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_MV_mask.nii');
            elif self.fusion_method_id == '1':
                command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_GWV_mask.nii');
            elif self.fusion_method_id == '2':
                command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_LWV_'+patch_radius+'_mask.nii');

        command.append('--patient-name');
        command.append(patient_name);
        command.append('--stack-name');
        command.append(prefix_stack_name);

        if segmentation_type == 'mono':
            command.append('--atlas-name');
            command.append(atlas_name);
            command.append('--output-csv');
            command.append(patient_mask_dir+'/mono_atlas_evaluation.csv');
        elif segmentation_type == 'fusion':
            command.append('--atlas-name');
            command.append('Fusion'+str(self.fusion_method_id));
            command.append('--output-csv');
            command.append(patient_mask_dir+'/multi_atlas_evaluation.csv');

        return command;

    def create_nuc_evaluation_command(self,patient_name,atlas_name,patient_mask_dir,patient_lr_path,segmentation_type,patch_radius):
        stack_name = patient_lr_path.split('/')[-1]
        prefix_stack_name = stack_name.split('_')[0]
        postfix_stack_name = stack_name.split('_')[1]

        command=[]
        command.append(self.mialsrtk_utilities_dir+'/mialsrtkEvaluateLabelOverlapMeasures');
        command.append('--ref');
        command.append(patient_mask_dir+'/'+prefix_stack_name+'_mask_'+postfix_stack_name);

        if segmentation_type == 'mono':
            command.append('--input');
            command.append(patient_mask_dir+'/AutoExtracted/'+atlas_name+'/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_lin_mask.nii');
        elif segmentation_type == 'fusion':
            command.append('--input');
            if self.fusion_method_id == '0':
                command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_NUCMV_mask.nii');
            elif self.fusion_method_id == '1':
                command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_NUCGWV_mask.nii');
            elif self.fusion_method_id == '2':
                command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_NUCLWV_'+patch_radius+'_mask.nii');

        command.append('--patient-name');
        command.append(patient_name);
        command.append('--stack-name');
        command.append(prefix_stack_name);

        if segmentation_type == 'mono':
            command.append('--atlas-name');
            command.append(atlas_name);
            command.append('--output-csv');
            command.append(patient_mask_dir+'/mono_atlas_evaluation.csv');
        elif segmentation_type == 'fusion':
            command.append('--atlas-name');
            command.append('NUCFusion'+str(self.fusion_method_id));
            command.append('--output-csv');
            command.append(patient_mask_dir+'/multi_atlas_evaluation.csv');

        return command;

    def create_nuc_old_ncc_evaluation_command(self,patient_name,atlas_name,patient_mask_dir,patient_lr_path,segmentation_type,patch_radius):
        stack_name = patient_lr_path.split('/')[-1]
        prefix_stack_name = stack_name.split('_')[0]
        postfix_stack_name = stack_name.split('_')[1]

        command=[]
        command.append(self.mialsrtk_utilities_dir+'/mialsrtkEvaluateLabelOverlapMeasures');
        command.append('--ref');
        command.append(patient_mask_dir+'/'+prefix_stack_name+'_mask_'+postfix_stack_name);

        if segmentation_type == 'mono':
            command.append('--input');
            command.append(patient_mask_dir+'/AutoExtracted/'+atlas_name+'/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_lin_mask.nii');
        elif segmentation_type == 'fusion':
            command.append('--input');
            if self.fusion_method_id == '0':
                command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_NUCMV_itkNCC_mask.nii');
            elif self.fusion_method_id == '1':
                command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_NUCGWV_itkNCC_mask.nii');
            elif self.fusion_method_id == '2':
                command.append(patient_mask_dir+'/AutoExtracted/'+str(".".join(patient_lr_path.split('/')[-1].split('.')[:-1]))+'_NUCLWV_itkNCC_'+patch_radius+'_mask.nii');

        command.append('--patient-name');
        command.append(patient_name);
        command.append('--stack-name');
        command.append(prefix_stack_name);

        if segmentation_type == 'mono':
            command.append('--atlas-name');
            command.append(atlas_name);
            command.append('--output-csv');
            command.append(patient_mask_dir+'/mono_atlas_evaluation.csv');
        elif segmentation_type == 'fusion':
            command.append('--atlas-name');
            command.append('NUCFusion'+str(self.fusion_method_id));
            command.append('--output-csv');
            command.append(patient_mask_dir+'/multi_atlas_evaluation_itkNCC.csv');

        return command;

    def browseOnClicked(self):
        print 'button browse clicked'
        self.data_directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Open Directory","/home/",QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks))
        print 'Data folder ( '+self.data_directory+' ) set'
        self.ui.data_lineEdit.setText(str(self.data_directory))

    def setDirOnClicked(self):
        print 'button set dir clicked'
        self.data_directory = str(self.ui.data_lineEdit.text())
        print 'directory '+str(self.data_directory)+' set'

    def setFusionOnClicked(self):
        if(self.ui.mv_radioButton.isChecked()):
            self.fusion_method_id = '0'
            self.ui.radius_label.setDisabled(True)
            self.ui.radius_spinBox.setDisabled(True)
            print 'Fusion method: Majority voting selected'
        elif(self.ui.gwv_radioButton.isChecked()):
            self.fusion_method_id = '1'
            self.ui.radius_label.setDisabled(True)
            self.ui.radius_spinBox.setDisabled(True)
            print 'Fusion method: Global weighted voting selected'
        elif(self.ui.lwv_radioButton.isChecked()):
            self.fusion_method_id = '2'
            self.ui.radius_label.setEnabled(True)
            self.ui.radius_spinBox.setEnabled(True)
            print 'Fusion method: Local weighted voting selected'

    def setPatchRadiusOnValueChanged(self):
        self.patch_radius = str(self.ui.radius_spinBox.value());
        print 'Patch radius set to ',self.patch_radius


if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
