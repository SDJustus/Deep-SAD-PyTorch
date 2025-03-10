import os
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms


class Visualizer():
    """ Visualizer wrapper based on Tensorboard.

    Returns:
        Visualizer: Class file.
    """
    def __init__(self, name):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join("../../tensorboard/Deep_SAD/", name))
        
    def denormalize(img):
        mean = np.array([0.4209137, 0.42091936, 0.42130423])
        std = np.array([0.34266332, 0.34264612, 0.3432589])
        img = (img * std + mean) * 255.0
        return img.astype(np.uint8)
        
    def plot_current_errors(self, total_steps, errors):
        """Plot current errros.

        Args:
            total_steps (int): Current total_steps
            errors (OrderedDict): Error for the current epoch.
        """
        self.writer.add_scalars("Loss", errors, global_step=total_steps)
        

    def plot_performance(self, epoch, performance):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            performance (OrderedDict): Performance for the current epoch.
        """
        
        self.writer.add_scalars("Performance Metrics", {k:v for k,v in performance.items() if (k != "conf_matrix" and k != "Avg Run Time (ms/batch)")}, global_step=epoch)
             
    def plot_current_conf_matrix(self, epoch, cm):
        
        def _plot_confusion_matrix(cm,
                          target_names=["Normal", "Abnormal"],
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          savefig = True):
            """
            given a sklearn confusion matrix (cm), make a nice plot

            Arguments
            ---------
            cm:           confusion matrix from sklearn.metrics.confusion_matrix

            target_names: given classification classes such as [0, 1, 2]
                        the class names, for example: ['high', 'medium', 'low']

            title:        the text to display at the top of the matrix

            cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                        see http://matplotlib.org/examples/color/colormaps_reference.html
                        plt.get_cmap('jet') or plt.cm.Blues

            normalize:    If False, plot the raw numbers
                        If True, plot the proportions

            Usage
            -----
            plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                    # sklearn.metrics.confusion_matrix
                                normalize    = True,                # show proportions
                                target_names = y_labels_vals,       # list of names of the classes
                                title        = best_estimator_name) # title of graph

            Citiation
            ---------
            http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

            """
            import matplotlib.pyplot as plt
            import numpy as np
            import itertools

            accuracy = np.trace(cm) / float(np.sum(cm))
            misclass = 1 - accuracy

            if cmap is None:
                cmap = plt.get_cmap('Blues')

            figure = plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            #plt.title(title)
            plt.colorbar()

            if target_names is not None:
                tick_marks = np.arange(len(target_names))
                plt.xticks(tick_marks, target_names, rotation=45)
                plt.yticks(tick_marks, target_names)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


            thresh = cm.max() / 1.5 if normalize else cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            if savefig:
                plt.savefig(title+".png")
            plt.close()
            return figure
        plot = _plot_confusion_matrix(cm, normalize=False, savefig=False)
        self.writer.add_figure("Confusion Matrix", plot, global_step=epoch)
        
    def plot_current_images(self, images, train_or_test="train", global_step=0, denormalize=False, device=None):
        """ Display current images.

        Args:
            global_step (int): global step
            train_or_test (["train", "test]): Determines, which phase the model is in
            images ([FloatTensor]): [Real Image, Anomaly Map, Mask (Optional)]
        """
        if denormalize:
            invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]), ])
            images = invTrans(images)
        self.writer.add_images("images from {} step: ".format(str(train_or_test)), images, global_step=global_step)