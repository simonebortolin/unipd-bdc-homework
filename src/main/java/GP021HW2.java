import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class GP021HW2 {

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
                .map(GP021HW2::strToVector)
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
        List<Vector> solution = seqWeightedOutliers(inputPoints,weights,k,z,0);

    }

    public static List<Vector> seqWeightedOutliers(ArrayList<Vector> inputPoints, ArrayList<Long> w, int k, int z, int alpha) {
        double r = getMin(inputPoints)/2;
        List<Integer> P = IntStream.range(0,inputPoints.size()).boxed().collect(Collectors.toList());


        while(r > 0.0 && r <= Double.MAX_VALUE) { // to prevent infinite loops
            List<Integer> Z = IntStream.range(0,inputPoints.size()).boxed().collect(Collectors.toList());
            List<Integer> S = new ArrayList<>(k);
            Long Wz = w.stream().reduce(Long::sum).orElse(0L);
            while(S.size() < k && Wz > 0) {
                double max = 0;
                Integer newCenter = null;
                for(int x : P) {
                    Long ballWeight = b(inputPoints, Z, x, (1+2*alpha)*r).stream().map(w::get).reduce(Long::sum).orElse(0L);
                    if(ballWeight > max) {
                        max = ballWeight;
                        newCenter = x;
                    }
                }
                if(newCenter != null) {
                    S.add(newCenter);
                    List<Integer> ball = b(inputPoints, Z, newCenter, (3+4*alpha)*r);
                    for(int y : ball) {
                        Z.remove((Integer)y);
                        Wz -= w.get(y);
                    }
                }
            }
            if(Wz <= z) {
                return S.stream().map(inputPoints::get).collect(Collectors.toList());
            } else {
                r *= 2;
            }
        }
        return null;
    }

    public static double getMin(ArrayList<Vector> inputPoints) {
        double d = Double.MAX_VALUE;
        int size = inputPoints.size();
        for(int i =0; i < size - 1; i++) {
            for(int j = i+1; j < size; j++) {
                d = Math.min(d,d(inputPoints.get(i), inputPoints.get(j)));
            }
        }
        return d;
    }

    public static double d(Vector x, Vector y) {
        return Math.sqrt(Vectors.sqdist(x, y));
    }

    public static List<Integer> b(List<Vector> inputPoints, List<Integer> Z, int x, double r) {
        return Z.stream().filter(y -> d(inputPoints.get(x), inputPoints.get(y)) <= r).collect(Collectors.toList());

    }
}
