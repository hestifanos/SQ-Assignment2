package org.example;
import java.io.File;
import java.io.IOException;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.DBSCAN;
import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;

public class ClusteringComparison {
    public static void main(String[] args) throws Exception {
        // Load the dataset
        CSVLoader loader = new CSVLoader();
        loader.setFile(new File("C:\\Users\\hermo\\Desktop\\SQLab22\\IDC\\src\\main\\java\\org\\example\\iris.csv"));
        Instances data = loader.getDataSet();

        // K-Means clustering
        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(3);
        kmeans.buildClusterer(data);

        // Hierarchical Agglomerative Clustering (HAC)
        HierarchicalClusterer hac = new HierarchicalClusterer();
        hac.setNumClusters(3);
        hac.buildClusterer(data);

        // DBSCAN clustering
        DBSCAN dbscan = new DBSCAN();
        dbscan.setEpsilon(0.3);
        dbscan.buildClusterer(data);

        // Evaluate clustering using silhouette score
        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(kmeans);
        eval.evaluateClusterer(data);
        System.out.println("K-Means Silhouette Score: " + eval.clusterResultsToString());

        eval.setClusterer(hac);
        eval.evaluateClusterer(data);
        System.out.println("HAC Silhouette Score: " + eval.clusterResultsToString());

        eval.setClusterer(dbscan);
        eval.evaluateClusterer(data);
        System.out.println("DBSCAN Silhouette Score: " + eval.clusterResultsToString());

        // Measure execution time
        long startTime = System.currentTimeMillis();
        kmeans.buildClusterer(data);
        long kmeansTime = System.currentTimeMillis() - startTime;
        System.out.println("K-Means Execution Time: " + kmeansTime + " ms");

        startTime = System.currentTimeMillis();
        hac.buildClusterer(data);
        long hacTime = System.currentTimeMillis() - startTime;
        System.out.println("HAC Execution Time: " + hacTime + " ms");

        startTime = System.currentTimeMillis();
        dbscan.buildClusterer(data);
        long dbscanTime = System.currentTimeMillis() - startTime;
        System.out.println("DBSCAN Execution Time: " + dbscanTime + " ms");
    }
}


