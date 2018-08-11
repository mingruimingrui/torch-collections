""" Sample training script for retinanet training on 1 GPU
Purpose of script is to demonstrate the expected inputs and outputs of
torch_collections.models.retinanet.RetinaNet
It is not intended to be optimized for training and recording of training statistics
For an optimized training script for retinanet on the coco dataset,
refer to TODO: MAKE SCRIPT
"""

import torch

from torch_collections.models.retinanet import RetinaNet


NUMBER_OF_CLASSES = 80
YOUR_DETECTION_DATASET_LOADER = None
TOTAL_STEPS = 500000
STEPS_PER_EPOCH = 500


def main():
    # Load dataset and create dataset loader
    ########## INSERT YOUR DATASET LOADER HERE ##########
    # dataset_loader should generate batches of data
    # A batch is expected to be a tuple. Here is what a batch would look like
    # batch = (
    #     'image'       : A torch.Tensor of images in the format NCHW
    #     'annotations' : A list of annotations for each image, list is of length N
    #                     Each annotation is a torch.Tensor of shape (number_of_annotations, 5)
    #                     and of format (x1, y1, x2, y2, class_id)
    # )
    dataset_loader = YOUR_DETECTION_DATASET_LOADER
    #####################################################

    # Load model with initial weights
    ########## INSERT NUMBER OF CLASSES HERE ##########
    retinanet = RetinaNet(num_classes=NUMBER_OF_CLASSES)
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
    ###################################################

    # Initialize optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, retinanet.parameters()), lr=0.00001)

    # Initialize training variables
    done = False  # done acts as a loop breaker
    count_steps = 0

    while not done:
        for image, annotations in dataset_loader:
            if torch.cuda.is_available():
                # Copy dataset to GPU
                image = image.cuda()
                annotations = [ann.cuda() for ann in annotations]

            # zero optimizer
            optimizer.zero_grad()

            # forward
            loss = retinanet(batch['image'], batch['annotations'])

            # In the event that loss is 0, None will be returned instead
            # This acts as a flag to signify that step can be skipped to
            # save the effort of backproping
            if loss is None:
                continue

            # backward + optimize
            loss.backward()
            optimizer.step()

            # Update training statistics
            count_steps += 1

            if count_steps % STEPS_PER_EPOCH == 0:
                # Record weights as latest
                torch.save(retinanet, 'snapshot/epoch_latest.pth')

            if count_steps >= TOTAL_STEPS:
                # Stop loop when required number of steps are passed
                done = True
                break

    torch.save(retinanet, 'snapshot/epoch_final.pth')
    print('Finished Training')


if __name__ == '__main__':
    # This step became required for python 3.X to share tensors for multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
