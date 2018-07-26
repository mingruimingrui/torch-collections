import torch

from torch_collections.models.retinanet import RetinaNet
from torch_collections.losses import DetectionFocalLoss, DetectionSmoothL1Loss


def main():
    # Load dataset
    #### INSERT YOUR DATASET HERE ####

    # dataset is a torch.utils.data.Dataset object
    # dataset should output samples of dict type with keys 'image' and 'annotations'
    # Here is what a sample would look like
    # sample = {
    #     'image'       : A numpy.ndarray image of dtype uint8 in RGB, HWC format and
    #     'annotations' : A numpy.ndarray of shape (number_of_annotations, 5)
    #                     Each annotation is of the format (x1, y1, x2, y2, class_id)
    # }
    dataset = YOUR_DETECTION_DATASET

    ##################################

    # Load model with initial weights
    #### INSERT NUMBER OF CLASSES HERE ####
    retinanet = RetinaNet(num_classes=NUMBER_OF_CLASSES)
    #######################################

    # Create dataset iterator
    collate_container = retinanet.build_collate_container()
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_container.collate_fn,
        num_workers=2,
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, retinanet.parameters()), lr=0.00001)

    for epoch in range(50):
        dataset_iterator = dataset_loader.__iter__()

        for step_nb in range(10000):
            # Get sample and zero optimizer
            batch = dataset_iterator.next()
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
