import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import os
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import cv2
from PIL import Image
import torchvision.transforms as transforms

writer = SummaryWriter('runs/G1G2')
SIZE=320
NC=14

def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256,192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256,192)

    return label_batch

def generate_label_color(inputs, opt):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], opt.label_nc))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)

    return input_label

def complete_compose(img,mask,label):
    label=label.cpu().numpy()
    M_f=label>0
    M_f=M_f.astype(np.int)
    M_f=torch.FloatTensor(M_f).cuda()
    masked_img=img*(1-mask)
    M_c=(1-mask.cuda())*M_f
    M_c=M_c+torch.zeros(img.shape).cuda()##broadcasting
    return masked_img,M_c,M_f

def compose(label,mask,color_mask,edge,color,noise):
    masked_label=label*(1-mask)
    masked_edge=mask*edge
    masked_color_strokes=mask*(1-color_mask)*color
    masked_noise=mask*noise
    return masked_label,masked_edge,masked_color_strokes,masked_noise

def changearm(old_label, data):
    label=old_label
    arm1=torch.FloatTensor((data['label'].cpu().numpy()==11).astype(np.int))
    arm2=torch.FloatTensor((data['label'].cpu().numpy()==13).astype(np.int))
    noise=torch.FloatTensor((data['label'].cpu().numpy()==7).astype(np.int))
    label=label*(1-arm1)+arm1*4
    label=label*(1-arm2)+arm2*4
    label=label*(1-noise)+noise*4
    return label

def generate_try_on(person_path, clothing_path):
    """Generate try-on result for a single person and clothing image"""
    try:
        # Initialize model and options
        opt = TrainOptions().parse()
        model = create_model(opt)
        
        # Create sample directory
        os.makedirs('sample', exist_ok=True)
        
        # Load and preprocess images
        person_img = Image.open(person_path).convert('RGB')
        clothing_img = Image.open(clothing_path).convert('RGB')
        
        # Resize images
        person_img = person_img.resize((256, 192))
        clothing_img = clothing_img.resize((256, 192))
        
        # Convert to tensors
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        person_tensor = transform(person_img).unsqueeze(0)
        clothing_tensor = transform(clothing_img).unsqueeze(0)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        person_tensor = person_tensor.to(device)
        clothing_tensor = clothing_tensor.to(device)
        
        # Generate masks
        mask_clothes = torch.FloatTensor((person_tensor.cpu().numpy() > 0).astype(np.int))
        mask_fore = torch.FloatTensor((person_tensor.cpu().numpy() > 0).astype(np.int))
        img_fore = person_tensor * mask_fore
        img_fore_wc = img_fore * mask_fore
        
        # Create data dictionary
        data = {
            'label': person_tensor,
            'edge': clothing_tensor,
            'image': person_tensor,
            'color': clothing_tensor,
            'pose': person_tensor,
            'name': [os.path.basename(person_path)]
        }
        
        # Generate try-on result
        with torch.no_grad():
            losses, fake_image, real_image, input_label, L1_loss, style_loss, clothes_mask, CE_loss, rgb, alpha = model(
                Variable(person_tensor.cuda()),
                Variable(clothing_tensor.cuda()),
                Variable(img_fore.cuda()),
                Variable(mask_clothes.cuda()),
                Variable(clothing_tensor.cuda()),
                Variable(changearm(person_tensor, data).cuda()),
                Variable(person_tensor.cuda()),
                Variable(person_tensor.cuda()),
                Variable(person_tensor.cuda()),
                Variable(mask_fore.cuda())
            )
        
        # Process result
        result = fake_image[0].cpu().numpy()
        result = (result + 1) / 2.0  # Denormalize
        result = (result * 255).astype(np.uint8)
        
        # Save result
        result_path = os.path.join('sample', f'result_{int(time.time())}.jpg')
        cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        return result_path
        
    except Exception as e:
        print(f"Error in generate_try_on: {str(e)}")
        return None

# Main training loop
if __name__ == '__main__':
    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
    else:    
        start_epoch, epoch_iter = 1, 0

    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# Inference images = %d' % dataset_size)

    model = create_model(opt)
    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    step = 0

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
            
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # Process batch
            t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float))
            mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
            mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(np.int))
            img_fore = data['image'] * mask_fore
            img_fore_wc = img_fore * mask_fore
            all_clothes_label = changearm(data['label'], data)

            # Forward pass
            losses, fake_image, real_image, input_label, L1_loss, style_loss, clothes_mask, CE_loss, rgb, alpha = model(
                Variable(data['label'].cuda()),
                Variable(data['edge'].cuda()),
                Variable(img_fore.cuda()),
                Variable(mask_clothes.cuda()),
                Variable(data['color'].cuda()),
                Variable(all_clothes_label.cuda()),
                Variable(data['image'].cuda()),
                Variable(data['pose'].cuda()),
                Variable(data['image'].cuda()),
                Variable(mask_fore.cuda())
            )

            # Process results
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + torch.mean(CE_loss)

            # Log results
            writer.add_scalar('loss_d', loss_D, step)
            writer.add_scalar('loss_g', loss_G, step)
            writer.add_scalar('loss_CE', torch.mean(CE_loss), step)
            writer.add_scalar('loss_g_gan', loss_dict['G_GAN'], step)

            # Save results
            a = generate_label_color(generate_label_plain(input_label), opt).float().cuda()
            b = real_image.float().cuda()
            c = fake_image.float().cuda()
            d = torch.cat([clothes_mask, clothes_mask, clothes_mask], 1)
            combine = torch.cat([a[0], d[0], b[0], c[0], rgb[0]], 2).squeeze()
            cv_img = (combine.permute(1,2,0).detach().cpu().numpy()+1)/2

            if step % 1 == 0:
                writer.add_image('combine', (combine.data + 1) / 2.0, step)
                rgb = (cv_img*255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite('sample/'+data['name'][0], bgr)

            step += 1
            print(step)

            if epoch_iter >= dataset_size:
                break

        # End of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        break

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        # np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
