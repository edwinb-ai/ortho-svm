package validacion;

import java.util.Random;
import msv.svm_node;
import msv.svm_problem;

/**
 * @author DOCTORADO-LCPG
 */
public class KFoldCrossValidator 
{

    /**
    * Algoritmo implementado por autor de Libsvm.
    * Stratified cross validation
    * @param prob
    * @param nr_fold 
     * @return  
    */
    public static svm_problem[] dividirProblema(svm_problem prob, int nr_fold)
    {
        int i;
        int[] fold_start = new int[nr_fold+1];
        int l = prob.l;
        int[] perm = new int[l]; //ÍNDICES DE LOS VECTORES EN EL DATASET ORDENADOS POR LA CLASE INDICADA EN tmp_label.
        Random rand = new Random();        
       
        if(nr_fold > l)
        {
            nr_fold = l;
            System.err.print("Warning: #Pliegues > #data. Se usará #Pliegues = #data en su lugar (i.e., leave-one-out cross validation\n");
        }
        
        // stratified cv may not give leave-one-out rate
        // Each class to l folds -> some folds may have zero elements
        if( nr_fold < l)
        {
            //LAS SIGUIENTES ESTRUCTURAS SIMULAN APUNTADORES A ARREGLOS.
            int[] tmp_nr_class = new int[1];
            int[][] tmp_label = new int[1][];
            int[][] tmp_start = new int[1][];
            int[][] tmp_count = new int[1][];

            svm_group_classes(prob,tmp_nr_class,tmp_label,tmp_start,tmp_count,perm);

            //RECUPERANDO LOS RESULTADOS DEL AGRUPAMIENTO DE CLASES.
            int nr_class = tmp_nr_class[0];
            int[] start = tmp_start[0]; //Posición del vector en el cuál inicia cada clase
            int[] count = tmp_count[0];	//Número de vectores en cada clase	            

            // random shuffle and then data grouped by fold using the array perm
            int[] fold_count = new int[nr_fold];
            int c;
            int[] index = new int[l];
            for(i=0;i<l;i++)
                index[i]=perm[i];
            
            //PERMUTANDO CADA ELEMENTO DE CADA CLASE CON ALGÚN OTRO DE LA MISMA CLASE ELEGIDO ALEATORIAMENTE.
            for (c=0; c<nr_class; c++)
                for(i=0;i<count[c];i++)
                {
                    int j = i+rand.nextInt(count[c]-i); 
                    do {int _t=index[start[c]+j]; index[start[c]+j]=index[start[c]+i]; index[start[c]+i]=_t;} while(false);
                }
            
            //CONTANDO LOS ELEMENTOS QUE TENDRÁ CADA UNO DE LOS k PLIEGUES.
            for(i=0;i<nr_fold;i++)
            {
                fold_count[i] = 0;
                for (c=0; c<nr_class; c++)
                    fold_count[i] += (i+1)*count[c]/nr_fold - i*count[c]/nr_fold;
            }
            
            //IDENTIFICANDO LA POSICIÓN DE LOS ELEMENTOS QUE MARCAN EL INICIO DE CADA PLIEGUE.
            fold_start[0]=0;
            for (i=1;i<=nr_fold;i++)
                fold_start[i] = fold_start[i-1] + fold_count[i-1];
            
            //COLOCA EN CADA PLIEGO LOS ELEMENTOS QUE LE CORRESPONDEN CLASE POR CLASE.
            for (c=0;c<nr_class;c++)
                for(i=0;i<nr_fold;i++)
                {
                    int begin = start[c]+i*count[c]/nr_fold; 
                    int end   = start[c]+(i+1)*count[c]/nr_fold;
                    for(int j=begin;j<end;j++)
                    {
                        perm[fold_start[i]] = index[j];
                        fold_start[i]++;
                    }
                }
            
            //REGRESANDO LOS INDICADORES DE PLIEGUE A SU ESTADO ORIGINAL.
            fold_start[0]=0;
            for (i=1;i<=nr_fold;i++)
                fold_start[i] = fold_start[i-1]+fold_count[i-1];
        }
        else //Número de pliegues = total de datos (l)
        {
            for(i=0;i<l;i++) perm[i]=i;
            for(i=0;i<l;i++) //Shuffling
            {
                int j = i+rand.nextInt(l-i);
                do {int _t=perm[i]; perm[i]=perm[j]; perm[j]=_t;} while(false);
            }
            for(i=0;i<=nr_fold;i++)
                fold_start[i]=i*l/nr_fold;
        }

        svm_problem[] subproblemas = new svm_problem[nr_fold];
        //REALIZANDO VALIDACIÓN CRUZADA.
        for(i=0;i<nr_fold;i++) 
        {
            int begin = fold_start[i];
            int end = fold_start[i+1];
            int j,k;
            
            svm_problem subprob2 = new svm_problem();

            subprob2.l = end-begin;
            subprob2.x = new svm_node[subprob2.l][];
            subprob2.y = new double[subprob2.l];
            
            //GUARDANDO LOS (begin-end) ELEMENTOS DE CADA PLIEGUE.
            k=0;
            for(j=begin;j<end;j++)
            { 
              subprob2.x[k] = prob.x[perm[j]];
              subprob2.y[k] = prob.y[perm[j]];
              ++k;
            }
            subproblemas[i] = subprob2;
        }
        //System.out.println("Perm: "+Arrays.toString(perm));
        //System.out.println("Start: "+Arrays.toString(fold_start));
        return subproblemas;
    }
    
    /**
    * IDENTIFICA Y CUENTA LAS CLASES QUE TIENE EL PROBLEMA (nr_class).
    * CONTABILIZA LOS ELEMENTOS DE CADA CLASE (count[]).
    * IDENTIFICA LOS MARCADORES DE CADA INICIO DE CLASE (start[]).
    * IDENTIFICA LAS ETIQUETAS DE CADA CLASE (count[]).
    * perm, length l, must be allocated before calling this subroutine.
    * @param prob
    * @param nr_class_ret
    * @param label_ret label: label name
    * @param start_ret start: begin of each class
    * @param count_ret count: #data of classes,
    * @param perm perm: indices to the original data
    */
    private static void svm_group_classes(svm_problem prob, int[] nr_class_ret, int[][] label_ret, int[][] start_ret, int[][] count_ret, int[] perm)
    {
        int l = prob.l;
        int max_nr_class = 16;
        int nr_class = 0;
        int[] label = new int[max_nr_class];
        int[] count = new int[max_nr_class];
        int[] data_label = new int[l];
        int i;

        for(i=0;i<l;i++)
        {
            int this_label = (int)(prob.y[i]);
            int j;
            for(j=0;j<nr_class;j++)
            {
                if(this_label == label[j])
                {
                    ++count[j];
                    break;
                }
            }
            data_label[i] = j;
            
            //Si se alcanza el máximo número de clases permitido, duplicar la capacidad.
            //Y además, copiar las etiquetas y contadores en un arreglo con la nueva capacidad
            if(j == nr_class)
            {       
                if(nr_class == max_nr_class)
                {
                    max_nr_class *= 2;
                    int[] new_data = new int[max_nr_class];
                    System.arraycopy(label,0,new_data,0,label.length);
                    label = new_data;
                    new_data = new int[max_nr_class];
                    System.arraycopy(count,0,new_data,0,count.length);
                    count = new_data;					
                }
                label[nr_class] = this_label;
                count[nr_class] = 1;
                ++nr_class;
            }
        }

        //
        // Labels are ordered by their first occurrence in the training set. 
        // However, for two-class sets with -1/+1 labels and -1 appears first, 
        // we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
        //
        if (nr_class == 2 && label[0] == -1 && label[1] == +1)
        {
            do {int _t=label[0]; label[0]=label[1]; label[1]=_t;} while(false);
            do {int _t=count[0]; count[0]=count[1]; count[1]=_t;} while(false);
            for(i=0;i<l;i++)
            {
                if(data_label[i] == 0)
                    data_label[i] = 1;
                else
                    data_label[i] = 0;
            }
        }

        int[] start = new int[nr_class];
      
        start[0] = 0;
        for(i=1;i<nr_class;i++)
            start[i] = start[i-1]+count[i-1];
        
        for(i=0;i<l;i++)
        {
            perm[start[data_label[i]]] = i;
            ++start[data_label[i]];
        }
        
        start[0] = 0;
        for(i=1;i<nr_class;i++)
            start[i] = start[i-1]+count[i-1];
        
        //Asignando los valores en los arreglos recibidos como parámetro. 
        //El 0 es por la simulación de apuntador
        nr_class_ret[0] = nr_class;
        label_ret[0] = label;
        start_ret[0] = start;
        count_ret[0] = count;
    }
}