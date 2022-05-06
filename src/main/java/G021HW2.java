import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class G021HW2 {

    static Container container;

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
        List<Integer> solution = seqWeightedOutliers(inputPoints,weights,k,z,0);
        Instant ends = Instant.now();
        double objective = computeObjective(inputPoints,solution,z);

        System.out.println("Input size n = "+inputPoints.size());
        System.out.println("Number of centers k = "+k);
        System.out.println("Number of outliers z = "+z);
        System.out.println("Initial guess = "+ container.getRs().get(0));
        System.out.println("Final guess = "+ container.getRs().get(container.getRs().size() - 1));
        System.out.println("Number of guesses = "+ container.getRs().size());
        System.out.println("Objective function = "+objective);
        System.out.println("Time of SeqWeightedOutliers = "+Duration.between(starts, ends).toMillis());

    }
    public static List<Integer> seqWeightedOutliers(ArrayList<Vector> inputPoints, ArrayList<Long> w, int k, int z, int alpha) {
        container = new Container(inputPoints);
        double r = getMinD(k+z+1)/2;
        container.getRs().add(r);
        List<Integer> P = IntStream.range(0,inputPoints.size()).boxed().collect(Collectors.toList());


        while(r > 0.0 && r <= Double.MAX_VALUE) { // to prevent infinite loops
            List<Integer> Z = IntStream.range(0,inputPoints.size()).boxed().collect(Collectors.toList());
            List<Integer> S = new ArrayList<>(k);
            Long Wz = w.stream().reduce(Long::sum).orElse(0L);
            while(S.size() < k && Wz > 0) {
                double max = 0;
                Integer newCenter = null;
                for(int x : P) {
                    Long ballWeight = b( Z, x, (1+2*alpha)*r).stream().map(w::get).reduce(Long::sum).orElse(0L);
                    if(ballWeight > max) {
                        max = ballWeight;
                        newCenter = x;
                    }
                }
                if(newCenter != null) {
                    S.add(newCenter);
                    List<Integer> ball = b( Z, newCenter, (3+4*alpha)*r);
                    for(int y : ball) {
                        Z.remove((Integer)y);
                        Wz -= w.get(y);
                    }
                }
            }
            if(Wz <= z) {
                return S;
            } else {
                r *= 2;
                container.getRs().add(r);
            }
        }
        return null;
    }

    public static double computeObjective(ArrayList<Vector> inputPoints, List<Integer> solution, int z) {
        ArrayList<Double> d = new ArrayList<>(inputPoints.size());
        for (int i =0; i< inputPoints.size(); i++) {
            double min = Double.MAX_VALUE;
            for (Integer j : solution) {
                min = Math.min(min, container.d(i, j));
            }
            d.add(min);
        }
        d.sort((a,b) -> -Double.compare(a,b));
        if(z < d.size())
            return d.get(z);

        return d.get(d.size() -1);
    }
    public static double getMinD(int size) {
        double d = Double.MAX_VALUE;
        for(int i =0; i < size - 1; i++) {
            for(int j = i+1; j < size; j++) {
                d = Math.min(d, container.d(i, j));
            }
        }
        return d;
    }

    public static List<Integer> b(List<Integer> Z, int x, double r) {
        return Z.stream().filter(y -> container.d(x,y) <= r).collect(Collectors.toList());
    }

    public static class Container {
        private final List<Double> rs = new ArrayList<>();

        private final List<Vector> inputPoints;
        private final Double[][] distance;

        public Container(List<Vector> inputPoints) {
            this.inputPoints = inputPoints;
            distance = new Double[inputPoints.size() -1][];
            for(int i =0; i < inputPoints.size() -1; i++) {
                distance[i] = new Double[inputPoints.size() - i - 1];
            }
        }
        public double d(int a, int b) {
            int i, j;
            if(a < b) {
                i = a;
                j = b-a-1;
            } else if (a > b) {
                i = b;
                j = a-b-1;
            } else return  0.0;

            if(distance[i][j] == null)
                distance[i][j] = Math.sqrt(Vectors.sqdist(inputPoints.get(a), inputPoints.get(b)));

            return distance[i][j];
        }

        public List<Vector> getInputPoints() {
            return inputPoints;
        }

        public List<Double> getRs() {
            return rs;
        }
    }
}
