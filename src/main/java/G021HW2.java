import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class G021HW2 {

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Input reading methods
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(G021HW2::strToVector)
                .forEach(result::add);
        return result;
    }

    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: <filename> <k> <z>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: <filename> <k> <z>");
        }

        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);

        ArrayList<Vector>  inputPoints = readVectorsSeq(filename);
        ArrayList<Long> weights = new ArrayList<>(Collections.nCopies(inputPoints.size(), 1L));
        Instant starts = Instant.now();
        List<Vector> solution = seqWeightedOutliers(inputPoints,weights,k,z,0);
        Instant ends = Instant.now();
        double objective = computeObjective(inputPoints,solution,z);

        System.out.println("Input size n = "+inputPoints.size());
        System.out.println("Number of centers k = "+k);
        System.out.println("Number of outliers z = "+z);
        System.out.println("Initial guess = "+ rs.get(0));
        System.out.println("Final guess = "+ rs.get(rs.size() - 1));
        System.out.println("Number of guesses = "+ rs.size());
        System.out.println("Objective function = "+objective);
        System.out.println("Time of SeqWeightedOutliers = "+Duration.between(starts, ends).toMillis());

    }
    public static List<Vector> seqWeightedOutliers(ArrayList<Vector> inputPoints, ArrayList<Long> w, int k, int z, int alpha) {
        Vector[] P = inputPoints.toArray(new Vector[0]);
        double r = getMinD(k+z+1, P)/2;
        rs.add(r);


        while(r > 0.0 && r <= Double.MAX_VALUE) { // to prevent infinite loops
            List<Vector> Z = new ArrayList<>(inputPoints);
            List<Long> Zw = new ArrayList<>(w);
            List<Vector> S = new ArrayList<>(k);
            long Wz = w.stream().reduce(Long::sum).orElse(0L);
            while(S.size() < k && Wz > 0) {
                double max = 0;
                Vector newCenter = null;
                for(int x = 0, size = P.length; x< size; x++) {
                    long ballWeight = cb(Z.toArray(new Vector[0]), Zw.toArray(new Long[0]),P[x], (1+2*alpha)*r);

                    if(ballWeight > max) {
                        max = ballWeight;
                        newCenter = P[x];
                    }
                }
                if(newCenter != null) {
                    S.add(newCenter);
                    long ballWeight = cbr( Z, Zw, newCenter, (3+4*alpha)*r);
                    Wz -= ballWeight;
                }
            }
            if(Wz <= z) {
                return S;
            } else {
                r *= 2;
                rs.add(r);
            }
        }
        return null;
    }

    public static double computeObjective(ArrayList<Vector> inputPoints, List<Vector> solution, int z) {
        ArrayList<Double> d = new ArrayList<>(inputPoints.size());
        for (int i =0; i< inputPoints.size(); i++) {
            double min = Double.MAX_VALUE;
            for (Vector j : solution) {
                min = Math.min(min, Math.sqrt(Vectors.sqdist(inputPoints.get(i), j)));
            }
            d.add(min);
        }
        d.sort((a,b) -> -Double.compare(a,b));
        if(z < d.size())
            return d.get(z);

        return d.get(d.size() -1);
    }
    public static double getMinD(int size, Vector[] inputPoints) {

        double d = Double.MAX_VALUE;
        for(int i =0; i < size - 1; i++) {
            for(int j = i+1; j < size; j++) {
                d = Math.min(d, Math.sqrt(Vectors.sqdist(inputPoints[i], inputPoints[j])));
            }
        }
        return d;
    }
    public static List<Double> rs = new ArrayList<>();

    private static long cb(Vector[] Z, Long[] W, Vector center, double v) {
        long weigth = 0;
        for(int i= 0, size = Z.length; i< size;i++) {
            if(Math.sqrt(Vectors.sqdist(center, Z[i])) <= v) {
                weigth += W[i];
            }
        }
        return weigth;
    }

    private static long cbr(List<Vector> Z, List<Long> W, Vector center, double v) {
        long weigth = 0;
        for(int i= Z.size() -1 ; i>=0;i--) {
            if(Math.sqrt(Vectors.sqdist(center, Z.get(i))) <= v) {
                weigth += W.get(i);
                W.remove(i);
                Z.remove(i);
            }
        }
        return weigth;
    }

}
