package msv;

/**@author Luis Carlos Padierna */
public class Gegenbauer
{
    public Gegenbauer(svm_parameter param)
    {
        if (param.degree < 0) System.out.println("El grado del polinomio debe ser mayor a 0");
    }

    public static double evaluate(svm_node[] x, svm_node[] y, int n, double alfa) 
    {
        if(n < 0) { System.out.println("Grado inválido. ¡Verifique!"); return  Double.MIN_VALUE ; }
        if(alfa==-0.5) return Chebyshev.evaluate(x, y, n);
        return K_geg(x, y, n, alfa);
    }
    
    /**
     * SUPONE QUE LOS VECTORES x e y TENDRÁN SIEMPRE EL MISMO NÚMERO DE COMPONENETES.
     * @param n Grado del polinomio.
     * @param y Vector escaso a evaluar.
     * @param x Vector escaso a evaluar. 
     * @param alfa 
     * @return  P_n(x) Evaluación de x en el polinomio de grado n.
     */
    public static double K_geg(svm_node[] x, svm_node[] y, int n, double alfa)
    {
        double sum, mult = 1;
        int xlen = x.length, ylen = y.length, i = 0, j = 0;
        
        while(i < xlen && j < ylen)
        {
            if(x[i].index == y[j].index)
            {
                sum = 1;
                for (int k = 1; k <= n; k++) 
                {
                    sum +=  G(x[i].value,k,alfa) * G(y[j].value,k,alfa) * w_1( x[i].value , y[j].value, alfa, k);
                }
                mult*=sum;
                //REGRESAR VALORES DE INFINITO AL MÁS GRANDE PERMITIDO
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
        //System.out.println(mult);
        return mult;
    }
            
    private static double w_1 (double x, double y, double alfa, int grado)
    {
        if (alfa <= 0) 
            return 1;
        else 
        {
            double inversoNormaCuadrada = pochhamer(2*alfa+1, grado)/pochhamer(1, grado);
            inversoNormaCuadrada = (inversoNormaCuadrada == 0) ? 1E-10 : 1/(inversoNormaCuadrada*inversoNormaCuadrada);
            return inversoNormaCuadrada *( Math.pow((1-x*x)*(1-y*y), alfa)+.1) / (grado+1); // 1/(grado +1) es útil para asegurar efecto de explosión.
        }                
    }
    
    /**
     * EVALÚA EL POLINOMIO LEGENDRE DE GRADO n PARA UN VALOR REAL.
     * @param x_i   valor.
     * @param n     grado.
     * @return      evaluación.
     */
    public static double G(double x_i, int n, double alfa)
    {        
        //CASO BASE
        switch (n) { case 0: return 1; case 1: return alfa*2*x_i; }
        //CASO RECURSIVO
        return  (1.0 / (n+1)) * ((2*(n+alfa)) * x_i * G(x_i,n-1,alfa)  -  (n+2*alfa-1) * G(x_i,n-2,alfa) );
    }
    
    /**
     * Implementa (x)_k tambien conocido como factorial desplazado.
     * (x)_k = Multiplicatoria desde (i=1) hasta (k) de: (x+i-1) para k = 1,2,3,...
     * (Dunkl and Xu, 2014, pp. 2)
     * @param x Factor de inicial
     * @param k Número de factores sucesivos que se multiplicarán por el factor inicial.
     * @return 
     */
    public static double pochhamer(double x, int k)
    {
        if(k == 0)  return 1.0;
        if(k < 0)   return 0.0;
        // k tiene prioridad sobre x: k se prueba primero.
        if(x == 0)  return 0.0;
        
        double aux = 1.0;
        for (int i = 0; i <= k-1; i++) 
            aux *= (x+i);
        return aux;
    }   
}