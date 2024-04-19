import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections

def triatpos(pos=(0,0), rot=0):
    r = np.array([[-1,-1],[1,-1],[1,1],[-1,-1]])*.5
    rm = [[np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot))],
           [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot))]]
    r = np.dot(rm, r.T).T
    r[:,0] += pos[0]
    r[:,1] += pos[1]
    return r

def triamatrix(a, annotations, ax, rot=0, cmap='GnBu', **kwargs):
    segs = []
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            segs.append(triatpos((j,i), rot=rot))
    col = collections.PolyCollection(segs, cmap=cmap, **kwargs)
    col.set_array(a.flatten())
    ax.add_collection(col)
    
    # Add annotations for im2 (triamatrix)
    if annotations is not None:
        for i, seg in enumerate(segs):
            x = np.mean(seg[:, 0])
            y = np.mean(seg[:, 1])
            ax.text(x, y, f'{annotations.flatten()[i]}', color='k', ha='center', va='top')
    
    return col


def custom_heatmap(data: pd.DataFrame, 
                   ycol: str, # column of the dataframe you want to put on the y axis
                   xcol: str, # column of the dataframe you want to put on the x axis of the heatmap,
                   ann_tiles: str = 'single', # split or single 
                   annotation_tile1: str = 'counts', # annotations inside the tiles: could be 'counts', 'proportions_col',  'proportions_row'
                   annotation_tile2: str = 'None',  # only one annotation by default; accepted: 'None', 'counts', 'proportions_col',  'proportions_row'
                   color_tiles: str = 'split', # split or single
                   color_tile1: str = 'proportions_col',  # color of the triangle1 based on 'counts', 'proportions_col',  'proportions_row',
                   color_tile2: str =  'proportions_row',  # color of the triangle2 based on 'counts', 'proportions_col',  'proportions_row',
                   lw = 0.5,  # linewidth parameter
                   ak: dict = {'fontsize': 4}, # annotation_kwas parameter
                   custom_cmap: str = 'autumn_r', 
                   custom_cmap2: str = 'summer_r',
                   figsize = (25, 25),  # Adjust the size of the heatmap
                   colorbar_width = 0.4,  # Adjust the width of the color bars
                   colorbar_pad = 0.05):
    '''Function to construct your heatmap to see correspondence between two columns (ycol, xcol). 
    You can choose to visualize the values of counts, proportions_col (where each column sums to 1), proportions_row (where each row sums to 1); 
    one can either visualize a combination of two or just one annotation. 
    For the color you can choose to use a single color or to splot the tile in two:'''
#################### define the annotations inside #####################
    if ann_tiles == 'single': 
        # define only annotation 1
        if annotation_tile1 == 'counts':
            annotation1 = data.groupby(ycol)[xcol].value_counts(normalize = False).unstack()   # counts
            annotation1.fillna(0, inplace=True)
            annotation1 = annotation1.round(0).astype(int)
        elif annotation_tile1 == 'proportions_row':
            annotation1 = data.groupby(ycol)[xcol].value_counts(normalize = True).unstack() # proportions row sum = 1
            annotation1 = annotation1.round(2)
        elif annotation_tile1 == 'proportions_col': 
            annotation1 = data.groupby(xcol)[ycol].value_counts(normalize = True).unstack().T # proportions col sum = 1
            annotation1 = annotation1.round(2)
        else: print(f'the value "{annotation_tile1}" that you specified is not accepted by the function. Please specify one from: counts, proportions_col, proportions_row')

    elif ann_tiles == 'split': 
        # define annotation 1
        if annotation_tile1 == 'counts':
            annotation1 = data.groupby(ycol)[xcol].value_counts(normalize = False).unstack()   # counts
            annotation1.fillna(0, inplace=True)
            annotation1 = annotation1.round(0).astype(int)
        elif annotation_tile1 == 'proportions_row':
            annotation1 = data.groupby(ycol)[xcol].value_counts(normalize = True).unstack() # proportions row sum = 1
            annotation1 = annotation1.round(2)
        elif annotation_tile1 == 'proportions_col': 
            annotation1 = data.groupby(xcol)[ycol].value_counts(normalize = True).unstack().T # proportions col sum = 1
            annotation1 = annotation1.round(2)
        else: print(f'the value "{annotation_tile1}" that you specified is not accepted by the function. Please specify one from: counts, proportions_col, proportions_row')
        # define annotation2
        if annotation_tile2 == 'counts':
            annotation2 = data.groupby(ycol)[xcol].value_counts(normalize = False).unstack()   # counts
            annotation1.fillna(0, inplace=True)
            annotation2 = annotation2.round(0).astype(int)
        elif annotation_tile2 == 'proportions_row':
            annotation2 = data.groupby(ycol)[xcol].value_counts(normalize = True).unstack() # proportions row sum = 1
            annotation2 = annotation2.round(2)
        elif annotation_tile2 == 'proportions_col': 
            annotation2 = data.groupby(xcol)[ycol].value_counts(normalize = True).unstack().T # proportions col sum = 1
            annotation2 = annotation2.round(2)
        else: print(f'the value "{annotation_tile2}" that you specified is not accepted by the function. Please specify one from: counts, proportions_col, proportions_row')
    else: 
        print(f'the value "{ann_tiles}" that you specified is not accepted by the function. Please specify one from: "split", "single"')

########################### define the colors ##################

    if color_tiles == 'single': 
        # define only annotation 1
        if color_tile1 == 'counts':
            color1 = data.groupby(ycol)[xcol].value_counts(normalize = False).unstack()   # counts
        elif color_tile1 == 'proportions_row':
            color1 = data.groupby(ycol)[xcol].value_counts(normalize = True).unstack() # proportions row sum = 1
        elif color_tile1 == 'proportions_col': 
            color1 = data.groupby(xcol)[ycol].value_counts(normalize = True).unstack().T # proportions col sum = 1
        else: print(f'the value "{color_tile1}" that you specified is not accepted by the function. Please specify one from: counts, proportions_col, proportions_row')

    elif color_tiles == 'split': 
        # define annotation1 
        if color_tile1 == 'counts':
            color1 = data.groupby(ycol)[xcol].value_counts(normalize = False).unstack()   # counts
        elif color_tile1 == 'proportions_row':
            color1 = data.groupby(ycol)[xcol].value_counts(normalize = True).unstack() # proportions row sum = 1
        elif color_tile1 == 'proportions_col': 
            color1 = data.groupby(xcol)[ycol].value_counts(normalize = True).unstack().T # proportions col sum = 1
        else: print(f'the value "{color_tile1}" that you specified is not accepted by the function. Please specify one from: counts, proportions_col, proportions_row')
        # define annotation2
        if color_tile2 == 'counts':
            color2 = data.groupby(ycol)[xcol].value_counts(normalize = False).unstack()   # counts
        elif color_tile2 == 'proportions_row':
            color2 = data.groupby(ycol)[xcol].value_counts(normalize = True).unstack() # proportions row sum = 1
        elif color_tile2 == 'proportions_col': 
            color2 = data.groupby(xcol)[ycol].value_counts(normalize = True).unstack().T # proportions col sum = 1
        else: print(f'the value "{color_tile2}" that you specified is not accepted by the function. Please specify one from: counts, proportions_col, proportions_row')
    
    else: 
        print(f'the value "{color_tiles}" that you specified is not accepted by the function. Please specify one from: "split", "single"')


    # Add print statements to check the type and values of annotation2

################################################## now plot ##########################################################################################
####################################################################################################################
    if ann_tiles == 'single' and color_tiles == 'single': 
                        fig, ax = plt.subplots(figsize=(figsize))
                        sns.heatmap(color1, annot = annotation1, linewidths = lw, annot_kws = ak, cmap = custom_cmap, ax = ax)
                        plot = plt.show()
        
    elif ann_tiles == 'single' and color_tiles == 'split': 
                # plot with colors1 ad color2 and only one annotation (annotation1)
                        fig, ax = plt.subplots(figsize = figsize)

                        # Plot im1 with no annotation
                        annotation1_val = annotation1.values
                        color1_val = color1.values        
                        annotation1_val = np.where(np.isnan(annotation1_val), '', annotation1_val.astype(str))
        
                        im1 = triamatrix(color1_val, annotation1_val, ax, rot=90, cmap=custom_cmap2)
                        im2 = ax.imshow(color2, cmap=custom_cmap)
        
                        # Add colorbars
                        cbar1 = fig.colorbar(im1, ax=ax, label= color_tile1, shrink=colorbar_width, pad=colorbar_pad)
                        cbar2 = fig.colorbar(im2, ax=ax, label= color_tile2, shrink=colorbar_width, pad=colorbar_pad)
                        cbar1.ax.yaxis.label.set_size(20)
                        cbar2.ax.yaxis.label.set_size(20)

                        xticklabels = color2.columns
                        yticklabels = color2.index

                        ax.set_xticks(np.arange(color1.shape[1] +0.5) , minor=False, ha=0.5)
                        ax.set_yticks(np.arange(color1.shape[0] +0.5), minor=False, va=0.5)
                        
                        cbar1.ax.tick_params(labelsize=12)  # Set the font size of color bar labels
                        cbar2.ax.tick_params(labelsize=12) 

                        x_min, x_max = ax.get_xlim()
                        y_min, y_max = ax.get_ylim()
                        
                        # Extend the x-axis by one unit to the right
                        ax.set_xlim(x_min, x_max - 0.5)
                        ax.set_ylim(y_min - 0.5, y_max)
        
                        ax.set_xlabel(xcol, fontsize=20)
                        ax.set_ylabel(ycol, fontsize=20)

                        title_text = f'Annotation in upper left tile: {annotation_tile2}; lower right tile: {annotation_tile1}'
                        ax.set_title(title_text, fontsize=22)
                     
                        plot = plt.show()      
        
    elif ann_tiles == 'split' and color_tiles == 'single': 
                # use only annotation1, annotation2, color1
                        fig, ax = plt.subplots(figsize=figsize)

                        annotation1_val = annotation1.values
                        annotation1_val = np.where(np.isnan(annotation1_val), '', annotation1_val.astype(str))
                        color1_val = color1.values
                        
                        im1 = triamatrix(color1_val, annotation1_val, ax, rot=90, cmap=custom_cmap)
                        im2 = ax.imshow(color1, cmap = custom_cmap)


                        if annotation_tile2 == 'counts': 
                            annotation2_numeric = annotation2.apply(pd.to_numeric, errors='coerce')
                            # Add annotations for im1 (imshow)
                            for i in range(annotation2_numeric.shape[0]):
                                for j in range(annotation2_numeric.shape[1]):
                                    annotation_text = str(annotation2_numeric.iat[i, j]).rstrip('0').rstrip('.')
                                    ax.text(j, i, annotation_text, color='k', ha='right', va='bottom')
                                    #ax.text(j, i, f'{annotation_text.iat[i, j]:.2f}', color='k', ha='right', va='bottom')
                        else: 
                            for i in range(annotation2.shape[0]):
                                  for j in range(annotation2.shape[1]):
                                      ax.text(j, i, f'{annotation2.iat[i, j]:.2f}', color='k', ha='right', va='bottom')     
      
                        # Add colorbars
                        cbar1 = fig.colorbar(im1, ax=ax, label= color_tile1, shrink=colorbar_width, pad=colorbar_pad)
                        cbar1.ax.yaxis.label.set_size(20)
                        cbar1.ax.tick_params(labelsize=12)

                        xticklabels = color1.columns
                        yticklabels = color1.index

                        ax.set_xticks(np.arange(color1.shape[1] +0.5) , minor=False, ha=0.5)
                        ax.set_yticks(np.arange(color1.shape[0] +0.5), minor=False, va=0.5)

                        x_min, x_max = ax.get_xlim()
                        y_min, y_max = ax.get_ylim()
                        
                        # Extend the x-axis by one unit to the right
                        ax.set_xlim(x_min, x_max - 0.5)
                        ax.set_ylim(y_min - 0.5, y_max)
        
                        ax.set_xlabel(xcol, fontsize=20)
                        ax.set_ylabel(ycol, fontsize=20)
                        title_text = f'Annotation in upper left tile: {annotation_tile1}; lower right tile: {annotation_tile2}'
                        ax.set_title(title_text, fontsize=22)
                        plot = plt.show()      
           
    elif ann_tiles == 'split' and color_tiles == 'split': 
                
                        fig, ax = plt.subplots(figsize=figsize)
                        # Plot im1 with colors from colors1
                        im1 = ax.imshow(color1, cmap=custom_cmap)
                        # Plot im2 with annotations from annotation2 
                        annotations2_val = annotation2.values
                        annotations2_val = np.where(np.isnan(annotations2_val), '', annotations2_val.astype(str))
                        color2_val = color2.values
                    
                        im2 = triamatrix(color2_val, annotations2_val, ax, rot=90, cmap=custom_cmap2)
                        
                        # Add colorbars
                        cbar1 = fig.colorbar(im1, ax=ax, label=color_tile1, shrink=colorbar_width, pad=colorbar_pad)
                        cbar2 = fig.colorbar(im2, ax=ax, label=color_tile2, shrink=colorbar_width, pad=colorbar_pad)
                        
                        cbar1.ax.yaxis.label.set_size(15)
                        cbar2.ax.yaxis.label.set_size(15)
        
                        cbar1.ax.tick_params(labelsize=12)  # Set the font size of color bar labels
                        cbar2.ax.tick_params(labelsize=12) 

                        if annotation_tile1 == 'counts': 
                            annotation1_numeric = annotation1.apply(pd.to_numeric, errors='coerce')
                            # Add annotations for im1 (imshow)
                            for i in range(annotation1_numeric.shape[0]):
                                for j in range(annotation1_numeric.shape[1]):
                                    annotation_text = str(annotation1_numeric.iat[i, j]).rstrip('0').rstrip('.')
                                    ax.text(j, i, annotation_text, color='k', ha='right', va='bottom')
                                    #ax.text(j, i, f'{annotation_text.iat[i, j]:.2f}', color='k', ha='right', va='bottom')
                        else: 
                            for i in range(annotation1.shape[0]):
                                  for j in range(annotation1.shape[1]):
                                      ax.text(j, i, f'{annotation1.iat[i, j]:.2f}', color='k', ha='right', va='bottom')      

                        # Set x-axis and y-axis tick labels
                        xticklabels = color1.columns
                        yticklabels = color1.index
                        # ax.set_xticklabels(xticklabels, rotation=90, ha='center')
                        # ax.set_yticklabels(yticklabels, va='center')

                        ax.set_xticks(np.arange(color1.shape[1] +0.5) , minor=False, ha=0.5)
                        ax.set_yticks(np.arange(color1.shape[0] +0.5), minor=False, va=0.5)
                        
                        x_min, x_max = ax.get_xlim()
                        y_min, y_max = ax.get_ylim()
                        
                        # Extend the x-axis by one unit to the right
                        ax.set_xlim(x_min, x_max - 0.5)
                        ax.set_ylim(y_min - 0.5, y_max)
        
                        ax.set_xlabel(xcol, fontsize=20)
                        ax.set_ylabel(ycol, fontsize=20)
                        title_text = f'Annotation in upper left tile: {annotation_tile1}; lower right tile: {annotation_tile2}'
                        ax.set_title(title_text, fontsize=22)
                        plot = plt.show()      

    else: print('error')
      

    return plot


