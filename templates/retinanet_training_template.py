""" Sample training script for retinanet training on 1 GPU, Multi GPU training
has not been implemented yet due to difficulties in securing resources
"""

import torch

from torch_collections.models.retinanet import RetinaNet


def main():
    # Load dataset and create dataset loader
    ########## INSERT YOUR DATASET LOADER HERE ##########
    # dataset_loader should generate batch of data
    # Here is what a batch would look like
    # batch = {
    #     'image'       : A torch.Tensor of images in the format NCHW
    #     'annotations' : A list of annotations for each image, list is of length N
    #                     Each annotation is a torch.Tensor of shape (number_of_annotations, 5)
    #                     and of format (x1, y1, x2, y2, class_id)
    # }
    dataset_loader = YOUR_DETECTION_DATASET_LOADER
    #####################################################

    # Load model with initial weights
    ########## INSERT NUMBER OF CLASSES HERE ##########
    retinanet = RetinaNet(num_classes=NUMBER_OF_CLASSES)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        retinanet = retinanet.to(device)
    ###################################################

    # Initialize optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, retinanet.parameters()), lr=0.00001)

    for epoch in range(50):
        dataset_iterator = dataset_loader.__iter__()

        for step_nb in range(len(dataset_loader)):
            # Get sample
            batch = dataset_iterator.next()
            if torch.cuda.is_available():

            # Zero optimizer
            optimizer.zero_grad()

            # forward
            loss = retinanet(batch)

            # In the event that loss is 0, None will be returned instead
            # This acts as a flag to signify that step can be skipped to
            # save the effort of backproping
            if loss is None:
                continue

            # backward + optimize
            loss.backward()
            optimizer.step()

        torch.save(retinanet, 'snapshot/epoch_{}.pth'.format(epoch))

    print('Finished Training')


if __name__ == '__main__':
    # This step became required for python 3.X to share tensors for multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
