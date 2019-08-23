# OHEM-loss
tensorflow  implementation of OHEM  loss  and Support the sigmoid or softmax entropy loss

# Instructions
 # softmax_ohem_loss
 cls_logits->[B, H, W, 2] # No softmax operation  
 cls_labels->[B, H, W]  
 train_mask->[B, H, W]  
 pcls_weights->[B, H, W]  
   
 cls_ohem = Ohem2(cls_logits,cls_labels,train_mask,negative_ratio=3.0)  
 cls_loss = cls_ohem.softmax_ohem_loss(pcls_weights)  
 
 # sigmoid_ohem_loss
 cls_logits->[B, H, W, 1] # No sigmoid operation  
 cls_labels->[B, H, W]  
 train_mask->[B, H, W]  
 pcls_weights->[B, H, W]  
   
 cls_ohem = Ohem2(cls_logits,cls_labels,train_mask,negative_ratio=3.0)  
 cls_loss = cls_ohem.sigmoid_ohem_loss(pcls_weights)  

