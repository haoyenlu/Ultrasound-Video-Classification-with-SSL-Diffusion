import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from configs.classification_config import Config, DiffusionClassifierConfig
from evaluate import Metric
from datetime import datetime

import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import transformers

from eval_utils import save_result_object, load_result_object

def alert_error(message: str = ""):
    if message == "": return
    messagebox.showwarning(title="Warning!" , message=message)

# ==================================================================
# User Interface for training model on myofascial pain classificaion 
# ==================================================================


class App(tk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master=master
        self.master.geometry("800x400")
        self.grid()
        


        ## Import Folder
        self.VIDEO_DIR = None
        self.import_folder_button = tk.Button(master, 
                                              text="Import Ultrasound Folder", 
                                              command=self.__import__folder
                                              )
        
        self.import_folder_label = tk.Label(master, text="", width=50)

        self.import_folder_button.grid(row=0,column=0,padx=10,pady=10)
        self.import_folder_label.grid(row=0,column=1,padx=10,pady=10)

        ## Import Patient CSV
        self.PATIENT_CSV = None
        self.import_file_button = tk.Button(master, 
                                            text="Import Patient CSV", 
                                            command=self.__import__patient_csv
                                            )
        
        self.import_file_label = tk.Label(master, text="", width=50)

        self.import_file_button.grid(row=1,column=0,padx=10,pady=10)
        self.import_file_label.grid(row=1,column=1,padx=10,pady=10)
        
        """Training Options"""
        ## Select Model
        model_options = ['Resnet50','Resnet101','vivit','VDE']
        self.select_model_label = tk.Label(master, text="Select Model:")
        self.select_model_label.grid(row=2,column=0,padx=10,pady=10, sticky='w')
        self.select_model_combo = ttk.Combobox(master, values=model_options,state='readonly')
        self.select_model_combo.current(0)
        self.select_model_combo.grid(row=2,column=0,padx=10, pady=10, sticky='e')

        ## Select Mode: Train or Test
        mode_options = ['Train','Test']
        self.select_mode_label = tk.Label(master, text="Select Mode:")
        self.select_mode_label.grid(row=2,column=1,padx=10, pady=10, sticky='w')
        self.select_mode_combo = ttk.Combobox(master, values=mode_options,state='readonly')
        self.select_mode_combo.current(0)
        self.select_mode_combo.grid(row=2,column=1,padx=10, pady=10, sticky='e')

        ## Training Parameters
        self.num_epoch_var = tk.IntVar(value=15)
        self.num_epoch_label = tk.Label(master, text='Numper of Epoch:').grid(row=3,column=0,padx=10,pady=10, sticky='w')
        self.num_epoch_entry = tk.Entry(master, width=10, textvariable=self.num_epoch_var).grid(row=3,column=0,padx=10,pady=10, sticky='e')
        
        self.learning_rate_var = tk.StringVar(value="0.0001")
        self.lr_label = tk.Label(master, text="Learning Rate:").grid(row=3,column=1,padx=10,pady=10, sticky='w')
        self.lr_entry = tk.Entry(master, width=10, textvariable=self.learning_rate_var).grid(row=3,column=1,padx=10,pady=10, sticky='e')

        self.use_demo_var = tk.BooleanVar()
        self.demo_checkbox = tk.Checkbutton(master, text='Use Demographic and Clinical Features', variable=self.use_demo_var)
        self.demo_checkbox.grid(row=4, column=0, padx=10,pady=10)
        
        self.repeat_cv_var = tk.IntVar(value="10")
        self.repeat_cv_label = tk.Label(master, text="Number of Repeat Cross-Validation:").grid(row=4,column=1,padx=10,pady=10, sticky='w')
        self.repeat_cv_entry = tk.Entry(master, width=10, textvariable=self.repeat_cv_var).grid(row=4,column=1,padx=10,pady=10, sticky='e')
        self.submit_button = tk.Button(master, text="Submit", command=self.__get__training_param, width=20).grid(row=5, column=0, columnspan=2, pady=10)




    
    
    def __import__folder(self):
        folder_path = filedialog.askdirectory(
            initialdir="./",
            title="Select Folder"
        )

        if folder_path:
            self.VIDEO_DIR = folder_path
            self.import_folder_label.config(text=f"{folder_path}")
    
    def __import__patient_csv(self):
        file_path = filedialog.askopenfilename(
            initialdir="./",
            title="Select csv File",
            filetypes=[("CSV files","*.csv")]
        )

        if file_path:
            self.PATIENT_CSV = file_path
            self.import_file_label.config(text=f"{file_path}")

    def __get__training_param(self):
        self.num_epoch = self.num_epoch_var.get()
        self.learning_rate = self.learning_rate_var.get()
        self.use_demo = self.use_demo_var.get()
        self.model = self.select_model_combo.get()
        self.mode = self.select_mode_combo.get()
        self.repeat_cv = self.repeat_cv_var.get()

        self.model_config = None
        if self.model == 'VDE':
            self.model_config = DiffusionClassifierConfig()
            if not self.use_demo: # not using demographic and clinical feature
                self.model_config.demo_param['sex_embedding'] = 0
                self.model_config.demo_param['age_embedding'] = 0
                self.model_config.demo_param['ppt_embedding'] = 0

        elif self.model == 'vivit':
            self.model_config = Config()
            self.model_config.model = self.model
            self.model_config.channel_first = False
        else:
            self.model_config = Config()
            self.model_config.model = self.model
            self.model_config.channel_first = True

        if self.model != 'VDE' and self.use_demo:
            alert_error("Only VDE model Supports Demographic and Clinical Feature!")
            return 
        

        self.model_config.num_epoch = int(self.num_epoch)
        self.model_config.learning_rate = float(self.learning_rate)
        self.model_config.mode = self.mode
        self.model_config.repeat_cross_validate = int(self.repeat_cv)

        curr_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model if not self.use_demo else self.model + '_demo'
        self.model_config.result_dir = './results/' + f"{model_name}_{curr_timestamp}"
        self.model_config.checkpoint_dir = './checkpoints/' +  f"{model_name}_{curr_timestamp}"


        if self.VIDEO_DIR is None or self.PATIENT_CSV is None:
            alert_error(message= "Should Import Ultrasound Video Folder and Patient CSV file first!")
            return 

        self.model_config.VIDEO_DIR = self.VIDEO_DIR
        self.model_config.PATIENT_CSV = self.PATIENT_CSV

        if self.mode == 'Train':
            trainer_window = TKTrainer(self.master, config=self.model_config)
        elif self.mode == 'Test':
            test_window = TKTester(self.master, config=self.model_config)

        else:
            alert_error(f"{self.mode} Not Supported!")
            return
        

from trainer import Trainer

class TKTrainer(Trainer):
    def __init__(self, master, config):
        super().__init__(config)
        self.master = master
        self.window = tk.Toplevel(master)
        self.window.geometry("800x600")

        self.start_btn = tk.Button(self.window, text="Start Training",command=self.train_start)
        self.start_btn.pack(pady=20)
        self.progress_bar = ttk.Progressbar(self.window, length=300, mode='determinate')
        self.progress_bar.pack(pady=20)
        self.fold_status = tk.Label(self.window, text="")
        self.fold_status.pack(pady=10)

        self.progress_bar_status = tk.Label(self.window, text="")
        self.progress_bar_status.pack(pady=10)

    
    
    def train_start(self):
        threading.Thread(target = self.train_cross_validation).start()
        self.start_btn.config(state='disabled')


    def train_finish(self):
        # Finished UI
        self.fold_status.config(text="Finished Training")
        self.progress_bar_status.config(text="")
        self.start_btn.config(state='normal', text='Start Testing')

    def train_cross_validation(self):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        for repeat in range(1,self.config.repeat_cross_validate+1):
            repeat_checkpoint_folder = f'CV-{repeat}'
            os.makedirs(os.path.join(self.config.checkpoint_dir,repeat_checkpoint_folder), exist_ok=True)

            for fold, test_patients in enumerate(self.specify_fold_test_patient.values(), 1):
                if fold not in self.config.folds_to_train: 
                    print('-' * 15, f"Skip Fold {fold}", '-' * 15)
                    continue
                
                # UI
                message = f"Repeat CV:{repeat}/{self.config.repeat_cross_validate} \n Training Fold:{fold}/4"
                self.fold_status.config(text=message)
                self.progress_bar["value"] = 0
                self.progress_bar_status.config(text="")

                test_ids  = [f"Patient {id}" for id in test_patients]
                train_ids = [pid for pid in self.patient_ids if pid not in test_ids]

                train_dataloader, test_dataloader = self._Trainer__get__dataloader(train_ids,test_ids)
                model = self._Trainer__load__model(fold=f"fold-{fold}")

                self.finetune(model , 
                              train_dataloader, 
                              test_dataloader, 
                              ckpt_path = os.path.join(self.config.checkpoint_dir, repeat_checkpoint_folder, f"fold-{fold}-checkpoint.pth") )
        

    
    def finetune(self, model, train_dataloader, test_dataloader, ckpt_path = 'checkpoint.pth'):
        if self.config.continue_training and os.path.exists(ckpt_path):
            print(f"Loading Checkpoint from {ckpt_path}!")
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt)

        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        best_model = None
        best_auc = 0.0


        train_metric , test_metric = Metric(), Metric()

        for epoch in range(self.config.num_epoch):
            ## Model Training
            model.train()
            for data in tqdm(train_dataloader):
                inputs, labels = data["clip"].to(self.device), data["label"].to(self.device).float().unsqueeze(-1)
                optimizer.zero_grad()

                if self.config.model == 'Diffusion':
                    outputs = model(data)

                else:
                    outputs = model(inputs)


                # handling huggingface output for training
                if isinstance(outputs, transformers.modeling_outputs.ImageClassifierOutput):
                    outputs = outputs.logits.float()
                

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                train_metric.update(y_true = labels.cpu().numpy(),y_pred = probs)


            ## Model Testing
            model.eval()
            with torch.no_grad():
                for data in tqdm(test_dataloader):
                    inputs = data["clip"].to(self.device)
                    if self.config.model == 'Diffusion':
                        outputs = model(data)

                    else:
                        outputs = model(inputs)

                    if isinstance(outputs, transformers.modeling_outputs.ImageClassifierOutput):
                        outputs = outputs.logits.float()

                    probs = torch.sigmoid(outputs).cpu().numpy()
                    test_metric.update(y_true = data["label"].numpy(),y_pred = probs)

            train_auc = train_metric.cal_auc_score()
            test_auc  = test_metric.cal_auc_score()

            current_progress = self.progress_bar_status.cget('text')
            new_progress = current_progress + '\n' + f"Epoch:{epoch}/{self.config.num_epoch}, Train AUC: {train_auc:.3f} , Test AUC: {test_auc:.3f}"
            self.progress_bar_status.config(text=new_progress)


            if test_auc > best_auc:
                best_auc = test_auc
                best_model = model.state_dict()
                torch.save(best_model,ckpt_path)

            train_metric.reset()
            test_metric.reset()

            self.progress_bar["value"] = int(((epoch + 1)/self.config.num_epoch) * 100)




class TKTester(Trainer):
    def __init__(self, master, config):
        super().__init__(config)

        self.master = master
        self.window = tk.Toplevel(master)
        self.window.geometry("800x600")

        self.checkpoint_dir = None
        ckpt_import_frame = tk.Frame(self.window)
        ckpt_import_frame.pack(pady=20)
        self.import_ckpt_button = tk.Button(ckpt_import_frame, 
                                              text="Import Checkpoint Folder", 
                                              command=self.__import__checkpoint)
        self.import_ckpt_button.pack(side='left', padx=10)
        
        self.import_ckpt_label = tk.Label(ckpt_import_frame, 
                                          text="", 
                                          width=50)
        self.import_ckpt_label.pack(side='right', padx=10)
        
        self.start_btn = tk.Button(self.window, text="Start Testing",command=self.test_start)
        self.start_btn.pack(pady=10)

        self.progress_bar = ttk.Progressbar(self.window, length=300, mode='determinate')
        self.progress_bar.pack(pady=20)

        self.results = None


    def __import__checkpoint(self):
        folder_path = filedialog.askdirectory(
            initialdir="./",
            title="Select Folder"
        )

        if folder_path:
            self.checkpoint_dir = folder_path
            self.import_ckpt_label.config(text=f"{folder_path}")
    
    


    def test_start(self):
        if self.checkpoint_dir is None:
            alert_error(message="Please Import Checkpoint Folder First!")
            return 
        
        self.working_thread = threading.Thread(target = self.test_cross_validation)
        self.working_thread.start()

        self.start_btn.config(state='disabled')
        self.check_thread()

    def check_thread(self):
        if self.working_thread.is_alive():
            self.window.after(100, self.check_thread)
        else:
            self.test_finish()

    
    def test_finish(self):
        from eval_utils import plot_roc_curves, save_result_object
        assert self.results is not None

        save_result_object(self.results, self.config.result_dir)

        figure = plot_roc_curves(self.results, self.config.model)
        self.result_canvas = FigureCanvasTkAgg(figure, master=self.window)
        self.result_canvas.draw()
        self.result_canvas.get_tk_widget().pack()


    def test_cross_validation(self):
        os.makedirs(self.config.result_dir)

        all_metric = []
        total_step = self.config.repeat_cross_validate * len(self.specify_fold_test_patient.keys())
        curr_step = 0

        for repeat in range(1,self.config.repeat_cross_validate+1):
            repeat_checkpoint_folder = f'CV-{repeat}'
            repeat_metric = []

            for fold , test_patients in enumerate(self.specify_fold_test_patient.values() , 1):                            
            
                test_ids  = [f"Patient {id}" for id in test_patients]
                train_ids = [pid for pid in self.patient_ids if pid not in test_ids]

                print(f"\nInference on Fold {fold} Test Samples",f"Test Patient:{','.join(test_ids)}")
                
                train_dataloader, test_dataloader = self._Trainer__get__dataloader(train_ids, test_ids)
                model = self._Trainer__load__model(fold=f"fold-{fold}")

                eval_metric = self.inference(model = model, 
                                             test_dataloader = test_dataloader, 
                                             ckpt_path = os.path.join(self.checkpoint_dir, repeat_checkpoint_folder , f"fold-{fold}-checkpoint.pth" ))

                repeat_metric.append(eval_metric)

                ## Progress BAR
                curr_step += 1
                self.progress_bar["value"] = int(((curr_step)/total_step) * 100)


            all_metric.append(repeat_metric)
        
        self.results = all_metric
    

    def inference(self, model , test_dataloader , ckpt_path):
        if os.path.exists(ckpt_path):
            print(f"Loading Checkpoint from {ckpt_path}!")
            if self.config.model == 'Diffusion':
                ckpt = torch.load(ckpt_path, map_location=self.device)
                new_state_dict = {k.replace("classifier_head.output_3d_layer.4.", "last."): v for k, v in ckpt.items()}
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(torch.load(ckpt_path))
        else:
            raise ValueError(f"{ckpt_path} does not exist @@")
        
        model.to(self.device)
        eval_metric = Metric(info=["patient_id","intervention"])
        model.eval()
        with torch.no_grad():
            for data in tqdm(test_dataloader):
                inputs = data["clip"].to(self.device)

                if self.config.model == 'Diffusion':
                    outputs = model(data)

                else:
                    outputs = model(inputs)

                if isinstance(outputs, transformers.modeling_outputs.ImageClassifierOutput):
                    outputs = outputs.logits.float()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                eval_metric.update(y_true=data["label"].numpy(), y_pred=probs)

                info = {"patient_id":data["patient"], "intervention": data["intervention"]}
                eval_metric.update_info(info) # get the patient of each clip
                
        return eval_metric





if __name__ == '__main__':

    root = tk.Tk()
    myapp = App(root)
    myapp.mainloop()