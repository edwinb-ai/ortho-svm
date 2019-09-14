package msv;

import javax.swing.JOptionPane;

/**
 * HOOSHAN 2016, IRAN,
 * Implementa la versión vectorial del Kernel Hermite propuesta en (Zhou et al, 2007)
 * Constructing Support Vector Machine Kernels from Orthogonal Polynomials for Face and Speaker Verification.
 * Fourth International Conference on Image and Graphics.
 * 11.nov.2015
 * @author Luis Carlos Padierna
 * @version $Id$
 */
public class Hermite 
{
    public Hermite(svm_parameter param)
    {
        if (param.degree < 0)
            JOptionPane.showMessageDialog(null,"El grado del polinomio debe ser mayor a 0","",JOptionPane.ERROR_MESSAGE);
    }

    public static double evaluate(svm_node[] x, svm_node[] y, int n) 
    {
        //VALIDANDO GRADO
        if(n < 0)
        {
            System.out.println("Grado inválido. ¡Verifique!");
            return  Double.MIN_VALUE ;
        }
        //VALIDANDO PRIMERA CONDICIÓN DE TAMAÑO. La segunda es que los índices de las componentes coincidan.
     //   if(x.length != y.length)
       //     return Double.NEGATIVE_INFINITY;
        double r = K_Her(x, y, n) ;
        //System.out.println("-> "+r);
        return r;
    }
    
    /**
     * Implementación del Kernel Hermite de HOOSHMAN 2016, (Zhou et al, 2012).
     * SUPONE QUE LOS VECTORES x e y TENDRÁN SIEMPRE EL MISMO NÚMERO DE COMPONENETES.
     * EN CASO QUE LOS VECTORES DIFIERAN EN TAMAÑO, NO SE OPERA SOBRE LAS COMPONENTES QUE NO TIENEN PAR.
     * @param n Grado del polinomio.
     * @param y Vector escaso a evaluar.
     * @param x Vector escaso a evaluar. 
     * @return  H_n(x) Evaluación de x en el polinomio de grado n.
     */
    public static double K_Her(svm_node[] x, svm_node[] y, int n)
    {
        double sum, mult = 1;
        int xlen = x.length, ylen = y.length, i = 0, j = 0;
        
        while(i < xlen && j < ylen)
        {
            if(x[i].index == y[j].index)
            {
                sum = 1;
                for (int k = 1; k <= n; k++) 
                    sum += H(x[i].value,k) * H(y[j].value,k) / (Math.pow(2, 2*n));
                mult*=sum;
                i++; j++;
            }
            else
            {
                if(x[i].index > y[j].index)
                    ++j;
                else
                    ++i;
            }
        }
        return mult;
    }
    
    
    
    /**
     * EVALÚA EL POLINOMIO HERMITE DE GRADO n PARA UN VALOR REAL.
     * @param x_i   valor.
     * @param n     grado.
     * @return      evaluación.
     */
    public static double H(double x_i, int n)
    {
        //double x = OperadoresBasicos.escalar(x_i,2);
        double x = x_i;
        //CASO BASE
        switch (n)
        {
            case 0: return 1;
            case 1: return x_i;
        }
        //CASO RECURSIVO
        return  x * H(x,n-1)  -  (n-1) * H(x,n-2);
    }
}