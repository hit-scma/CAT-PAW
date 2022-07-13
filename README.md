# CAT-PAW

Due to the limitation of GitHub on file size, we uploaded the model file to Google drive. You can download and use it according to the instructions below.

DExperts model link: [DExperts models](https://drive.google.com/drive/folders/1SQyu_QEIHAT7SoQDz2qdR5ojGgMW56Ml?usp=sharing)    
GeDi model link: [GeDi models](https://drive.google.com/drive/folders/1FFxZh04KHaaTuwo-99B5pCO2Gzw9U7VM?usp=sharing)    

Usage:  
You can copy the files stored in Google Drive File Path to Project File Path under the root directory of "papercode/" after downloading according to the links.
  
Google Drive File Path  | Project File Path
 ---- | ------  
 DExperts/sentiment/positive/pytorch_model.bin  | DExperts/experts/sentiment/large/finetuned_gpt2_positive/pytorch_model.bin 
 DExperts/sentiment/negative/pytorch_model.bin  | DExperts/experts/sentiment/large/finetuned_gpt2_negative/pytorch_model.bin 
 DExperts/toxicity/nontoxic/pytorch_model.bin  | DExperts/experts/toxicity/large/finetuned_gpt2_nontoxic/pytorch_model.bin 
 DExperts/toxicity/toxic/pytorch_model.bin  | DExperts/experts/toxicity/large/finetuned_gpt2_toxic/pytorch_model.bin 
 GeDi/detoxifier/pytorch_model.bin  | GeDi/pretrained_models/gedi_detoxifier/ pytorch_model.bin
 GeDi/sentiment/pytorch_model.bin  | GeDi/pretrained_models/gedi_sentiment/ pytorch_model.bin 
 GeDi/topic/pytorch_model.bin  | GeDi/pretrained_models/gedi_topic/ pytorch_model.bin 
