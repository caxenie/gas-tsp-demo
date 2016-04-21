/*
    Sample demo software for Genetic Algorithms Exercise
    Traveling Salesman Problem Solution using GAs.
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

/* GAs parameters */
#define POPULATION_SIZE     50            // chromosomes
#define MAX_GENERATIONS     1000          // number of generations to evolve
#define XOVER_PROB          0.8           // crossover probability
#define MUTATION_PROB       0.3          // mutation probability
#define MAX_CHROMOSOME_SIZE 1000

/* gene abstraction */
struct gene{
    /* each gene represents a point in the TSP problem */
    /* id and position in the map */
    int id;
    int x;
    int y;
};

/* chromosome abstraction */
struct chromosome{
    /* size of the chromosome in genes */
    int csize;
    /* a chromosome is build of genes */
    struct gene *genes;
    /* the fitness value of the chromosome - the total distance for the chromosome genes */
    double fitness;
    double cfitness;
    double rfitness;
};

/* chromosomes population */
struct population{
    /* size in chromosomes */
    int size;
    /* chromosomes in the population */
    struct chromosome *c;
    /* current evolution generation */
    int gen;
    /* index of the fittest chromosome */
    int best_chromosome_idx;
};

/* The problem assumes that the GA should find the smallest distance when going
  from one point back to itself and passing through each other point once.
*/

int chromosome_size  = 0;
int min = 0;
int max = 0;
int** dataset;
int dataset_len = 0;
int rand_seed = 0;

/* random value generator within the problem bounds for the cities coordinates */
int randomize_pos(int l, int h)
{
    return rand()%(h - l) + l;
}

/* initialize a gne */
void init_gene(struct gene *g, int i, int x, int y){
    g->id = i;
    g->x = x;
    g->y = y;
}

/* shuffle the genes when initializing chromosomes */
int* shuffle_chromosome(int *data, int n)
{
    int* ret = (int*)calloc(n, sizeof(int));
    if (n > 1) {
        for (int i = 0; i < n - 1; i++) {
            int j = i + 1 + (rand() + rand_seed) % (n - i - 1);
            int t = data[j];
            data[j] = data[i];
            data[i] = t;
        }
    }
    for (int i = 0; i < n; i++) {
        ret[i] = data[i];
    }

    /* make the shuffling reliable */
    rand_seed+=POPULATION_SIZE;

    return ret;
}

/* initialize a chromosome */
void init_chromosome(struct chromosome *c, int s){
    int *genes_ids = (int*)calloc(s, sizeof(int));
    struct gene* gene_data = (struct gene*)calloc(s, sizeof(struct gene));

    for(int i=0;i<s;++i){
        genes_ids[i] = dataset[i][0];
    }
    /* number of genes */
    c->csize = s;

    /* allocate and init genes */
    c->genes = (struct gene*)calloc(c->csize, sizeof(struct gene));
    for(int i=0;i<c->csize;++i){
        init_gene(&c->genes[i], dataset[i][0], dataset[i][1], dataset[i][2]);
    }
    /* save a copy of the input data set into an aux gene representation for index - coord matching */
    for(int i=0;i<s;i++){
        gene_data[i].id = c->genes[i].id;
        gene_data[i].x = c->genes[i].x;
        gene_data[i].y = c->genes[i].y;
    }

     int* shuffled_genes = shuffle_chromosome(genes_ids, s);

    for(int i=0;i<c->csize;i++){
        /* ensure that although we are shuffling ids we keep the correct coord values in */
        c->genes[i].id = shuffled_genes[i];
        c->genes[i].x = gene_data[shuffled_genes[i]].x;
        c->genes[i].y = gene_data[shuffled_genes[i]].y;
    }

    /* init fitness value */
    c->fitness = 0.0f;
    c->rfitness = 0.0f;
    c->cfitness = 0.0f;
    free(genes_ids);
    free(shuffled_genes);
    free(gene_data);
}

/* initialize a chromosomes population with given parameters */
void init_population(struct population *p, int psize){
    p->size = psize;
    p->gen = 0;
    p->best_chromosome_idx = 0;
    p->c = (struct chromosome*)calloc(p->size+1, sizeof(struct chromosome));
    for(int i=0; i<p->size+1; i++){// include also the best chromosome at the end
        init_chromosome(&p->c[i], chromosome_size);
    }
}

/* computes the fitness of a chromosome in the population */
double compute_fitness(struct chromosome *c){
    double distanceQuadraticSum = 0.0f;
    for(int i=0;i<c->csize-1;++i){
        distanceQuadraticSum += (pow(c->genes[i].x - c->genes[i+1].x, 2) + pow(c->genes[i].y - c->genes[i+1].y, 2));
    }
    double fitness_val = 1.0/ sqrt(distanceQuadraticSum);
    return fitness_val;
}

/* evaluate function, takes a user defined function and computes it for every chromosome */
void evaluate_population(struct population *p)
{
    for(int i=0; i<p->size; i++){
        p->c[i].fitness = compute_fitness(&p->c[i]);
    }
}

/* select the best (fittest) chromosome in the population */
void select_best(struct population *p)
{
  // double min = DBL_MAX;
    double maxFitness = 0;
    p->best_chromosome_idx = 0;

    for(int i=0; i<p->size; ++i){
        /* the last entry in the population is the best chromosome */
        if (maxFitness < p->c[i].fitness){
            maxFitness = p->c[i].fitness;
            p->best_chromosome_idx = i;
        }
    }
    p->c[POPULATION_SIZE].fitness = maxFitness;

    /* found the fittest then copy the genes */
    for(int i=0;i<p->c[p->best_chromosome_idx].csize;++i){
        p->c[POPULATION_SIZE].genes[i].id = p->c[p->best_chromosome_idx].genes[i].id;
        p->c[POPULATION_SIZE].genes[i].x = p->c[p->best_chromosome_idx].genes[i].x;
        p->c[POPULATION_SIZE].genes[i].y = p->c[p->best_chromosome_idx].genes[i].y;
    }
}

/* apply elitism so that if the previous best chromosome is better than the
 * current generation best the first will replace the worst chromosome in the
 * current generation.
 */
void apply_elitism(struct population *p)
{
    struct chromosome *best = (struct chromosome*)calloc(1, sizeof(struct chromosome));
    struct chromosome *worst= (struct chromosome*)calloc(1, sizeof(struct chromosome));
    int best_idx = 0, worst_idx = 0;
    init_chromosome(best, chromosome_size);
    init_chromosome(worst, chromosome_size);
    best->fitness = p->c[0].fitness;
    worst->fitness = p->c[0].fitness;

    for(int i=0;i< p->size-1;++i){
        if(p->c[i].fitness < p->c[i+1].fitness){
            if(p->c[i].fitness <= best->fitness){
                best->fitness = p->c[i].fitness;
                best_idx = i;
            }
            if(p->c[i+1].fitness >= worst->fitness){
                worst->fitness = p->c[i+1].fitness;
                worst_idx = i+1;
            }
        }
        else{
            if(p->c[i].fitness >= worst->fitness){
                worst->fitness = p->c[i].fitness;
                worst_idx = i;
            }
            if(p->c[i+1].fitness <= best->fitness){
                best->fitness = p->c[i+1].fitness;
                best_idx = i+1;
            }
        }
    }
    /* if best chromosome from the new population is better than */
    /* the best chromosome from the previous population, then    */
    /* copy the best from the new population; else replace the   */
    /* worst chromosome from the current population with the     */
    /* best one from the previous generation                     */
    if(best->fitness <= p->c[POPULATION_SIZE].fitness){
        for(int i=0;i<p->c[best_idx].csize;++i){
            p->c[POPULATION_SIZE].genes[i].id = p->c[best_idx].genes[i].id;
            p->c[POPULATION_SIZE].genes[i].x = p->c[best_idx].genes[i].x;
            p->c[POPULATION_SIZE].genes[i].y = p->c[best_idx].genes[i].y;
        }
        p->c[POPULATION_SIZE].fitness = p->c[best_idx].fitness;
    }
    else{
        for(int i=0;i<p->c[worst_idx].csize;++i){
            p->c[worst_idx].genes[i].id = p->c[POPULATION_SIZE].genes[i].id;
            p->c[worst_idx].genes[i].x = p->c[POPULATION_SIZE].genes[i].x;
            p->c[worst_idx].genes[i].y = p->c[POPULATION_SIZE].genes[i].y;
        }
        p->c[worst_idx].fitness = p->c[POPULATION_SIZE].fitness;
    }
}

/* selection function using the elitist model in which only the
 * best chromosomes survive
 */
void apply_selection(struct population *p, struct population *newp)
{
    double sum_fit = 0.0f;
    double prob = 0.0f;
    /* find the global value of the fitness of the population */
    for(int i=0; i< p->size; ++i){
        sum_fit+=p->c[i].fitness;
    }
    /* compute the relative fitness of the population */
    for(int i=0; i<p->size; ++i){
        p->c[i].rfitness = p->c[i].fitness/sum_fit;
    }
    p->c[0].cfitness = p->c[0].rfitness;

    /* compute the cumulative fitness of the population */
    for(int i=1; i< p->size; ++i){
        p->c[i].cfitness = p->c[i-1].cfitness + p->c[i].rfitness;
    }
    /* select the survivors using the cumulative fitness */
    for(int i=0;i<p->size;++i){
        prob = rand()%1000/1000.0;
        if(prob > p->c[0].cfitness)
            newp->c[i] = p->c[0];
        else
        {
            for(int j=0; j<p->size; ++j){
                if(prob<=p->c[j].cfitness && prob>p->c[j+1].cfitness)
                    newp->c[i] = p->c[j+1];
            }
        }
    }
    /* one the new population is created copy it back in the working var */
    for(int i=0 ;i<p->size; ++i)
        p->c[i] = newp->c[i];
}

/* apply the single point crossover operator which takes 2 parents */
void apply_crossover(struct population *p)
{
    /* counter of members chosen */
    int cnt = 0;
    /* probability to xover */
    double prob_xover = 0.0f;
    /* the two parent containers init */
    struct chromosome *p1 = (struct chromosome*)calloc(1, sizeof(struct chromosome));
    init_chromosome(p1, chromosome_size);
    /* crossover loop */
    for(int i=0; i< p->size; ++i){
        prob_xover = rand()%1000/1000.0;
        if(prob_xover < XOVER_PROB){
            cnt++;
            if(cnt%2==0){
                for(int j=0;j<p->c[i].csize;++j){
                    double tmpid, tmpx, tmpy;
                    tmpid = p1->genes[j].id;
                    tmpx = p1->genes[j].x;
                    tmpy = p1->genes[j].y;

                    p1->genes[j].id = p->c[i].genes[j].id;
                    p1->genes[j].x = p->c[i].genes[j].x;
                    p1->genes[j].y = p->c[i].genes[j].y;

                    p->c[i].genes[j].id = tmpid;
                    p->c[i].genes[j].x = tmpx;
                    p->c[i].genes[j].y = tmpy;
                }
            }
        }
        else
        {
            p1 = &p->c[i];
        }
    }
}
/* function to swap 2 genes in the chromosome */
void swap_genes(struct chromosome *c, int idx1, int idx2){
    int tmpid = 0, tmpx = 0, tmpy = 0;
    int found1 = 0, found2 = 0;
    for(int i=0;i<c->csize;i++){
        if(c->genes[i].id==idx1) found1 = i;
        if(c->genes[i].id==idx2) found2 = i;
    }
    tmpid = c->genes[found1].id;
    tmpx = c->genes[found1].x;
    tmpy = c->genes[found1].y;

    c->genes[found1].id = c->genes[found2].id;
    c->genes[found1].x = c->genes[found2].x;
    c->genes[found1].y = c->genes[found2].y;


    c->genes[found2].id = tmpid;
    c->genes[found2].x = tmpx;
    c->genes[found2].y = tmpy;
}

/* apply mutation - random uniform mutation of the genes */
void apply_mutation(struct population *p)
{
    double prb = 0.0f;
    for(int i=0;i<p->size;++i){
        for(int j=0;j<p->c[i].csize;++j){
            prb = rand()%1000/1000.0;
            if(prb < MUTATION_PROB){
                int rand_id = randomize_pos(min, max);
                int cur_id = p->c[i].genes[j].id;
                if(rand_id != cur_id){
                    swap_genes(&p->c[i], cur_id, rand_id);
                }
            }
        }
    }
}


/* print the state of the current evolution generation */
void report_state(struct population *p)
{
    printf("Generation: %d | Best fitness: %lf\n", p->gen, p->c[POPULATION_SIZE].fitness);
}

/* get the input data and store it locally for the population initialization */
void get_input_dataset(char *  filename) {
     FILE *fp = fopen(filename, "r");
     if (fp == NULL){
          printf("Fail to open file %s, %s.\n", filename, strerror(errno));
          return;
     }
    printf("Open file %s OK.\n", filename);
    int id = 0, x = 0, y = 0;
    int input_idx = 0;

    dataset = (int **)calloc(MAX_CHROMOSOME_SIZE, sizeof(int*));
    for(int i=0;i<MAX_CHROMOSOME_SIZE;++i){
        dataset[i] = (int *)calloc(3, sizeof(int));
    }
    /* loop and get training data */
    while (fscanf(fp,"%d,%d,%d", &id, &x, &y) != EOF){
        /* populate the training set vector*/
        dataset[input_idx][0] = id;
        dataset[input_idx][1] = x;
        dataset[input_idx][2] = y;
        /* check if max input reached */
        if(input_idx==1000) break;
        input_idx++;
    }
    fclose(fp);
    dataset_len = input_idx;
    chromosome_size  = dataset_len;
    min = 1;
    max = chromosome_size;
}

/* entry point */
int main(int argc, char* argv[]){
    srand(time(NULL));
    printf("\n\nSimulation for GAs started...\n\n");
    struct population *p = (struct population*)calloc(1, sizeof(struct population));
    struct population *newp = (struct population*)calloc(1, sizeof(struct population));
    get_input_dataset(argv[1]);
    init_population(p, POPULATION_SIZE);
    init_population(newp, POPULATION_SIZE);
    /* init from the input dataset */
    evaluate_population(p);
    select_best(p);
    report_state(p);
    while(p->gen < MAX_GENERATIONS ){
        p->gen++;
        apply_selection(p, newp);
        apply_crossover(p);
        apply_mutation(p);
        report_state(p);
        evaluate_population(p);
        apply_elitism(p);
    }
    printf("\nEvolution is completed...\n\n");
    printf("\nBest chromosome:\n");
    for(int i=0;i<p->c[POPULATION_SIZE].csize;++i){
        printf(" %d ", p->c[POPULATION_SIZE].genes[i].id);
    }
    printf("\n");

    printf("\n Best fitness: %lf\n\n", p->c[POPULATION_SIZE].fitness);
    printf("\nSimulation ended.\n\n");
    free(p);
    free(newp);
    free(dataset);
    return EXIT_SUCCESS;
}

