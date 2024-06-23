use plotters::prelude::*;
use rand::prelude::*;

type Point = [f32; 2];

struct KMeansData {
    points: Vec<Point>,
    k: usize,
    centroids: Vec<Point>,
}
struct ResultKMeans {
    clusters: Vec<Vec<Point>>,
    centroids: Vec<Point>,
}

fn create_data(center: Point, range: f32, num_points: usize) -> Vec<Point> {
    // Generamos puntos aleatorios para usar luego aplicar en el algoritmo
    let mut data: Vec<Point> = vec![[0.0, 0.0]; num_points];
    let mut rng: ThreadRng = rand::thread_rng();
    for i in 0..num_points {
        let x = rng.gen_range(-range * 0.5..range * 0.5);
        let y = rng.gen_range(-range * 0.5..range * 0.5);
        data[i] = [center[0] + x, center[1] + y];
    }
    data
}

fn euclidean_distance(point1: Point, point2: Point) -> f32 {
    // Implementación de la fórmula de distancia entre dos puntos en este caso en dos dimensiones
    let sq_diff_x = (point1[0] - point2[0]).powi(2);
    let sq_diff_y = (point1[1] - point2[1]).powi(2);
    (sq_diff_x + sq_diff_y).sqrt()
}

fn group_clusters(
    k: usize,
    kmeans_data: &KMeansData,
    assignments: Vec<usize>,
) -> Vec<Vec<[f32; 2]>> {
    let mut output: Vec<Vec<Point>> = Vec::new();
    // Como ya sabemos el valor de k, evitamos realocación de memoria innecesaria prealocando K clusters dentro del vector output
    for _ in 0..k {
        output.push(vec![[0.0, 0.0]; 2]);
    }

    output.extend(std::iter::repeat(vec![[0.0, 0.0]; 2]).take(k));
    for (i, point) in kmeans_data.points.iter().enumerate() {
        let cluster = assignments[i];
        output[cluster].push(*point);
    }
    output
}

fn check_convergence(kmeans_data: &KMeansData, new_centroids: &Vec<Point>) -> bool {
    // Comprueba si los centroids de los K clusters son iguales a los calculados en la iteración actual
    if kmeans_data
        .centroids
        .iter()
        .zip(new_centroids)
        .any(|(old, new)| old[0] != new[0] || old[1] != new[1])
    {
        return false;
    }
    return true;
}

fn update_centroids(kmeans_data: &KMeansData, temp_assignments: &Vec<usize>) -> Vec<Point> {
    // Creamos dos vectores de tamaño K rellenos de puntos (0,0):
    // new_centroids va a guardar los puntos a los cuales considere centroids en esta iteracion.
    // counts es un vector que guarda el cluster al que pertenece cada punto en este ciclo.
    let mut new_centroids: Vec<Point> = vec![
        kmeans_data
            .points
            .iter()
            .map(|_| [0.0, 0.0])
            .next()
            .unwrap();
        kmeans_data.k
    ];
    let mut counts: Vec<f32> = vec![0.0; kmeans_data.k];
    // En el for loop sumamos todos los puntos de cada cluster
    // y sumamos la cantidad de puntos que tiene cada cluster
    for (i, point) in kmeans_data.points.iter().enumerate() {
        let cluster = temp_assignments[i];
        new_centroids[cluster][0] += point[0];
        new_centroids[cluster][1] += point[1];
        counts[cluster] += 1.0;
    }
    // Usando .map() sacamos el promedio de cada cluster usando: Suma de puntos / Cant. de puntos
    new_centroids
        .iter()
        .enumerate()
        .map(|(i, point)| point.map(|x| x / counts[i]))
        .collect()
}

fn assign_points(kdata: &KMeansData) -> Vec<usize> {
    // Un vector temporal que almacena a que cluster le pertenece cada punto
    let mut temp_assignment: Vec<_> = Vec::with_capacity(kdata.points.len());
    // Buscamos por distancia euclidiana a que centroid esta mas cerca el punto
    for point in &kdata.points {
        let mut index = 0;
        let mut min_distance = euclidean_distance(*point, kdata.centroids[0]);
        for (i, centroid) in kdata.centroids.iter().enumerate().skip(1) {
            let distance = euclidean_distance(*point, *centroid);
            if distance < min_distance {
                // Si la distancia es menor, asignamos a ese centroid como "ganador"
                min_distance = distance;
                index = i;
            }
        }
        temp_assignment.push(index);
    }
    temp_assignment
}

fn kmeans_alg(data: Vec<Point>, k: usize) -> ResultKMeans {
    // Inicializamos el struct + asignamos puntos aleatorios como centroids (inicializacion estandar del algoritmo)
    // "The Forgy method randomly chooses k observations from the dataset and uses these as the initial means" https://en.wikipedia.org/wiki/K-means_clustering
    let mut kmeans_data = KMeansData {
        centroids: (0..k)
            .map(|_| {
                *data
                    .choose(&mut rand::thread_rng())
                    .expect("Error al asignar los centroids")
            })
            .collect(),
        k,
        points: data.clone(),
    };
    // Nos sirve guardar el vector final de assignments para luego agruparlo y graficarlo
    let mut convergence = false;
    let mut assignments = Vec::with_capacity(data.len());
    while !convergence {
        // Asignar Puntos
        let temp_assignments: Vec<usize> = assign_points(&kmeans_data);

        //Actualizar Centroids
        let new_centroids = update_centroids(&kmeans_data, &temp_assignments);

        // Comprobar convergencia
        convergence = check_convergence(&kmeans_data, &new_centroids);
        if convergence {
            assignments = temp_assignments;
        }
        kmeans_data.centroids = new_centroids;
    }
    // Una vez se consiguen los centroids y a que cluster pertenece cada punto, agrupamos en un Vec<[[f32;2]; k]>
    ResultKMeans {
        clusters: group_clusters(k, &kmeans_data, assignments),
        centroids: kmeans_data.centroids,
    }
}

fn main() {
    let data_array1: Vec<Point> = create_data([7.0, 5.0], 1.0, 150); // Cluster 1
    let data_array2: Vec<Point> = create_data([2.0, 2.0], 2.0, 150); // Cluster 2
    let data_array3: Vec<Point> = create_data([3.0, 8.0], 1.0, 150); // Cluster 3
    let data_array4: Vec<Point> = create_data([5.0, 5.0], 9.0, 300); // Ruido
    let data: Vec<Point> = [data_array1, data_array2, data_array3, data_array4].concat();

    let kmeans = kmeans_alg(data, 3);

    // Construimos el gráfico de puntos

    let root = BitMapBackend::new("target/kmeans.png", (600, 400)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let x_lim = 0.0..10.0f32;
    let y_lim = 0.0..10.0f32;

    let mut ctx = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 40) // Put in some margins
        .set_label_area_size(LabelAreaPosition::Right, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("KMeans", ("sans-serif", 15)) // Set a caption and font
        .build_cartesian_2d(x_lim, y_lim)
        .expect("Error al construir el gráfico");

    ctx.configure_mesh().draw().unwrap();

    for (cluster_i, cluster) in kmeans.clusters.iter().enumerate() {
        let color = match cluster_i {
            // adjust for more colors
            0 => &BLUE,
            1 => &RED,
            2 => &GREEN,
            3 => &YELLOW,
            4 => &MAGENTA,
            5 => &CYAN,
            _ => &BLACK,
        };
        //let style = PointStyle(color, 5);
        let points_iter = cluster.iter().map(|point| (point[0], point[1]));
        ctx.draw_series(PointSeries::of_element(
            points_iter.clone(),
            3,
            color,
            &|c, s, st| {
                return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
                + Circle::new((0,0),s,st.filled()); // At this point, the new pixel coordinate is established
            },
        ))
        .expect("Error al dibujar los puntos.");
        ctx.draw_series(PointSeries::of_element(
            kmeans
                .centroids
                .iter()
                .map(|centroid| (centroid[0], centroid[1])),
            5,
            &BLACK,
            &|c, s, st| {
                return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
            + Circle::new((0,0),s,st.filled()); // At this point, the new pixel coordinate is established
            },
        ))
        .expect("Error al dibujar los centroids.");
    }
}
