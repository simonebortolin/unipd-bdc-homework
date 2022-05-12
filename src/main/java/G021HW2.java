import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class G021HW2 {

    private static final List<Double> rs = new ArrayList<>();

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
    public static List<Vector> seqWeightedOutliers(ArrayList<Vector> ip, ArrayList<Long> weights, int k, int z, int alpha) {
        Vector[] inputPoints = ip.toArray(new Vector[0]);
        //Long[] w = weights.toArray(new Long[0]);

        double r = getMinD(k+z+1, inputPoints)/2;
        rs.add(r);
        HashMap<Vector, Long> P = new HashMap<>();
        for(int i : IntStream.range(0,ip.size()).boxed().collect(Collectors.toList())) {
            P.put(inputPoints[i], weights.get(i));
        }

        while(r > 0.0 && r <= Double.MAX_VALUE) { // to prevent infinite loops
            HashMap<Vector, Long> Z = new HashMap<>(P);
            List<Vector> S = new ArrayList<>(k);
            long Wz = Z.values().stream().reduce(Long::sum).orElse(0L);
            while(S.size() < k && Wz > 0) {
                double max = 0;
                Vector newCenter = null;
                for(Map.Entry<Vector, Long> x : P.entrySet()) {
                    Map<Vector, Long> ball = cb( Z, x.getKey(), (1+2*alpha)*r);
                    long ballWeight = ball.values().stream().reduce(Long::sum).orElse(0L);
                    if(ballWeight > max) {
                        max = ballWeight;
                        newCenter = x.getKey();
                    }
                }
                if(newCenter != null) {
                    S.add(newCenter);
                    Map<Vector, Long>  ball = cb( Z, newCenter, (3+4*alpha)*r);
                    ball.keySet().forEach(Z::remove);
                    long ballWeight = ball.values().stream().reduce(Long::sum).orElse(0L);
                    Wz -=ballWeight;
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

    private static List<Integer> cb(List<Integer> Z, int center, double r, Vector[] inputPoints) {
        List<Integer> list = new ArrayList<>(Z.size());
        for(int i= 0 , size = Z.size(); i< size;i++) {
            if(Math.sqrt(Vectors.sqdist(inputPoints[i], inputPoints[center])) <= r) {
                list.add(Z.get(i));
            }
        }

        return list;
    }

    private static Map<Vector, Long> cb(Map<Vector, Long> inputPoints, Vector center, double r) {
        Map<Vector, Long> map = new HashMap<>();
        for(Map.Entry<Vector, Long> item : inputPoints.entrySet()) {
            if(Math.sqrt(Vectors.sqdist(item.getKey(), center)) <= r) {
                map.put(item.getKey(), item.getValue());
            }
        }
        return map;
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


}