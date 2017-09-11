import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, silhouette_samples, silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

files_directory = './data'

SEED = 16121993

'''
Dummy variables transformed into variables with
real values.

This rule was followed:
1 = Legit
0 = Suspicious
-1 = Phising
Source: http://eprints.hud.ac.uk/id/eprint/19220/3/RamiPredicting_Phishing_Websites_based_on_Self-Structuring_Neural_Network.pdf 
'''
def dummy_to_labels(df, result_included=True, stat_report_included=True):
    df['URL_Length'] = df['URL_Length'].apply(lambda x: 'lt_54' if x == 1 else 'gte_54_lte_75' if x == 0 else 'gt_75')

    df['having_Sub_Domain'] = df['having_Sub_Domain'].apply(lambda x: 'no_subd' if x == 1 else 'one_subd' if x == 0 else 'multi_subd')

    df['SSLfinal_State'] = df['SSLfinal_State'].apply(lambda x: 'https&issuer' if x == 1 else 'https' if x == 0 else 'nothing')

    df['URL_of_Anchor'] = df['URL_of_Anchor'].apply(lambda x: 'anchor%_lt_31' if x == 1 else 'anchor%_gte_31_lte_67' if x == 0 else 'anchor%_gt_67')

    df['Links_in_tags'] = df['Links_in_tags'].apply(lambda x: 'links%_lt_17' if x == 1 else 'links%_gte_17_lte_81' if x == 0 else 'links%_gt_81')

    df['SFH'] = df['SFH'].apply(lambda x: 'other' if x == 1 else 'dif_domain' if x == 0 else 'blank')

    df['web_traffic'] = df['web_traffic'].apply(lambda x: 'alexa_lt_100K' if x == 1 else 'alexa_gt_100K' if x == 0 else 'not_in_alexa')

    df['Links_pointing_to_page'] = df['Links_pointing_to_page'].apply(lambda x: 'gt_2' if x == 1 else '1_or_2' if x == 0 else 'zero')

    df['having_IP_Address'] = df['having_IP_Address'].apply(lambda x: 'false' if x == 1 else 'true')

    df['Shortining_Service'] = df['Shortining_Service'].apply(lambda x: 'false' if x == 1 else 'true')

    df['having_At_Symbol'] = df['having_At_Symbol'].apply(lambda x: 'false' if x == 1 else 'true')

    df['double_slash_redirecting'] = df['double_slash_redirecting'].apply(lambda x: 'false' if x == 1 else 'true')

    df['Prefix_Suffix'] = df['Prefix_Suffix'].apply(lambda x: 'false' if x == 1 else 'true')

    df['Domain_registeration_length'] = df['Domain_registeration_length'].apply(lambda x: 'gt_1_year' if x == 1 else 'lte_1_year')

    df['Favicon'] = df['Favicon'].apply(lambda x: 'same_domain' if x == 1 else 'external_domain')

    df['port'] = df['port'].apply(lambda x: 'not_preferred_status' if x == 1 else 'preferred_status')

    df['HTTPS_token'] = df['HTTPS_token'].apply(lambda x: 'false' if x == 1 else 'true')

    df['Request_URL'] = df['Request_URL'].apply(lambda x: 'requestUrl%_lt_22' if x == 1 else 'requestUrl%_gt_61')

    df['Submitting_to_email'] = df['Submitting_to_email'].apply(lambda x: 'other' if x == 1 else 'mail_or_mailto')

    df['Abnormal_URL'] = df['Abnormal_URL'].apply(lambda x: 'hostname_in_url' if x == 1 else 'no_hostname_in_url')

    df['Redirect'] = df['Redirect'].apply(lambda x: '0_or_1' if x == 1 else '2_or_3')

    df['on_mouseover'] = df['on_mouseover'].apply(lambda x: 'no_statusbar_change' if x == 1 else 'statusbar_change')

    df['RightClick'] = df['RightClick'].apply(lambda x: 'enabled' if x == 1 else 'disabled')

    df['popUpWidnow'] = df['popUpWidnow'].apply(lambda x: 'other' if x == 1 else 'yes&with_form')

    df['Iframe'] = df['Iframe'].apply(lambda x: 'false' if x == 1 else 'true')

    df['age_of_domain'] = df['age_of_domain'].apply(lambda x: 'gte_6_months' if x == 1 else 'lt_6_months')

    df['DNSRecord'] = df['DNSRecord'].apply(lambda x: 'found' if x == 1 else 'not_found')

    df['Page_Rank'] = df['Page_Rank'].apply(lambda x: 'gt_0.2' if x == 1 else 'lt_0.2')

    df['Google_Index'] = df['Google_Index'].apply(lambda x: 'true' if x == 1 else 'false')

    if stat_report_included:
        df['Statistical_report'] = df['Statistical_report'].apply(lambda x: 'legit' if x == 1 else 'phising')

    if result_included:
        df['Result'] = df['Result'].apply(lambda x: 'legit' if x == 1 else 'phising')

    return df

'''
This function assigns real values to the dummy values 
present in the dataset when dummy=True
'''
def wrangle_data(file_name='Training_Dataset.csv', dummy=True):
    df = pd.read_csv(files_directory + '/'+ file_name)

    df.drop(['id'], axis=1, inplace=True) 

    if not dummy:
        df = dummy_to_labels(df)

    return df

'''
Performance report for categorical data 
given an array of real values and an array of predicted values.
'''
def predicted_report(y_test, y_pred):
    print('%s\n' % pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True))

    print('Accuracy: %f\n' % accuracy_score(y_test, y_pred))

    print(classification_report(y_test, y_pred))


'''
Auxiliar function to show the results of SkLearn Cross-Validation Grid-Search in a fancy way.
'''
def k_folds_evaluation(validation_results):
    print('Accuracies: ')
    print(validation_results['test_accuracy'])
    print('Mean accuracy: ')
    print('%0.3f (+/- %0.3f)' % (np.mean(validation_results['test_accuracy']), np.std(validation_results['test_accuracy'])))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('F1 values: ')
    print(validation_results['test_f1'])
    print('Mean F1: ')
    print('%0.3f (+/- %0.3f)' % (np.mean(validation_results['test_f1']), np.std(validation_results['test_f1'])))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Precisions: ')
    print(validation_results['test_precision'])
    print('Mean precision: ')
    print('%0.3f (+/- %0.3f)' % (np.mean(validation_results['test_precision']), np.std(validation_results['test_precision'])))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Recalls: ')
    print(validation_results['test_recall'])
    print('Mean recall: ')
    print('%0.3f (+/- %0.3f)' % (np.mean(validation_results['test_recall']), np.std(validation_results['test_recall'])))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

'''
The following function was taken from this site: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
'''
def silhouette_plots(range_n_clusters, X, y):
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=SEED, n_jobs=-1, init='random')
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()
