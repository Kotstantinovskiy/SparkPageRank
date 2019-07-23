import com.google.common.collect.Lists;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.ArrayList;

public class PageRank {
    public static void main(String[] args){
        SparkConf sparkConf = new SparkConf();
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
        JavaRDD<String> rddInput = sparkContext.textFile(Config.livejournal).cache();
        rddInput = rddInput.filter(x -> !x.startsWith("#"));

        JavaPairRDD<Integer, ArrayList<Integer>> graph = JavaPairRDD.fromJavaRDD(rddInput.map(x->{
            String[] parts = x.split("\t");
            return new Tuple2<Integer, Integer>(Integer.valueOf(parts[0]), Integer.valueOf(parts[1]));
        })).groupByKey()
                .mapToPair(x -> new Tuple2<>(x._1(), Lists.newArrayList(x._2())))
                .cache();

        graph = graph.flatMapToPair(x -> {
            ArrayList<Tuple2<Integer, Integer>> arrayList_tmp = new ArrayList<>();
            for(int v : x._2()){
                arrayList_tmp.add(new Tuple2<>(v, -1));
                arrayList_tmp.add(new Tuple2<>(x._1(), v));
            }
            return arrayList_tmp;
        }).groupByKey().mapToPair(x ->{
            boolean flag = true;
            ArrayList<Integer> arrayList_tmp = new ArrayList<>();
            for(int v : Lists.newArrayList(x._2())){
                if(v != -1){
                    flag = false;
                    arrayList_tmp.add(v);
                }
            }

            if(flag){
                arrayList_tmp.add(-1);
                return new Tuple2<>(x._1(), arrayList_tmp);
            } else {
                return new Tuple2<>(x._1(), arrayList_tmp);
            }
        }).cache();

        long vertexCount = graph.count();
        long hangCount = graph.filter(x -> x._2().size() == 1 && x._2().get(0) == -1).count();
        double startPR = 1 / (double) vertexCount;

        JavaPairRDD<Integer, Double> pageRank = graph.mapToPair(x -> new Tuple2<>(x._1(), startPR));

        for(int i=0; i < Config.ITERATIONS; i++){
            double hangPR = graph.join(pageRank).filter(x -> x._2()._1().size() == 1 && x._2()._1().get(0) == -1).map(x -> x._2()._2()).reduce((x, y) -> x + y);
            hangPR = hangPR / hangCount;
            double finalHangPR = hangPR;

            pageRank = graph.join(pageRank).flatMapToPair(x -> {
                    int len = x._2()._1().size();
                    double pageRank_tmp = x._2()._2();
                    pageRank_tmp = pageRank_tmp / len;
                    ArrayList<Tuple2<Integer, Double>> arrayList_tmp = new ArrayList<>();

                    for(int v : x._2()._1()){
                        if(v != -1) {
                            arrayList_tmp.add(new Tuple2<>(v, pageRank_tmp));
                        }
                    }

                    return arrayList_tmp;
                }).reduceByKey((x, y) -> {
                    if(x == 0){
                        x = x + Config.ALPHA * (1 / vertexCount);
                        x = x + (1-Config.ALPHA) * finalHangPR;
                    }
                    return x + y;
            });
        }

        pageRank.mapToPair(x -> x.swap()).sortByKey(false).mapToPair(x -> x.swap()).saveAsTextFile(String.valueOf(args[0]));
    }
}
