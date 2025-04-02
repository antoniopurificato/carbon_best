import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_three_images(img_paths, titles=None, general_title=None, figsize=(15, 5), save_path=None):
    assert len(img_paths) == 3, 
    if titles:
        assert len(titles) == 3, 

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for i, img_path in enumerate(img_paths):
        img = mpimg.imread(img_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        if titles:
            axes[i].set_title(titles[i], fontsize=12)

    if general_title:
        plt.suptitle(general_title, fontsize=16, y=1.05)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    plt.tight_layout()
    plt.show()


img_paths = ["filter_pareto_42_foursquare_tky_perc0_new.png", "filter_pareto_42_foursquare_tky_perc30_new.png", "filter_pareto_42_foursquare_tky_perc70_new.png"]
title = "CIFAR-10"

plot_three_images(img_paths, save_path="foursquare_42_all_fronts")
