## Practice 4 performed with Fibonacci sequence 

1.- Version recursiva descendente  

def fibonacci1(n:Int): Int = 
{
    if (n<2)
    {
        return n
    }
    else
    {
        return n
    }
    else 
    {
        return (fibonacci(n-1) + fibonacci(n-2))
    }
}

2.- Version con formula explicita  

var p: Double = 0;
var j: Double = 0;

def fibonacci2(n:Double): Double =
{
    if (n<2)
    {
        return n
    }
    else 
    {
        p = ((1+ Math.sqrt(5))/2)
        j=((Math.pow(p,n)- Math.pow((1-p),n))/ Math.sqrt(5))

        return j
    }
}

3.-Version interativa  

def fibonacci3(n3:Int):Int=
{
    var a=0;
    var b=0;
    var c=0;

    for (k <- Range (0,n3))
    {
        c=b+a
        a=b
        b=c
    }
    return a
}

4.- Version iterativa 2 variables    

def fibonacci4(n:Int):Int=
{
    var a=0;
    var b=1;

    for(k <- Range (0,n))
    {
        b=b+a;
        a=b-a;
    }

    return b;
}

5.- Version iterativa con vector     

def fibonacci5(n:Int): Int=
{
    if(<-2)
    {
        return n;
    }
    else
    {
        var vect = Array.ofDim[Int](n+1);

        vect(0)=0;
        vect(1)=1;

        for(k<- Range(2,2+1))
        {
            vect(k)=aglo(k-1)+ vect(k-2);
        }
        return vect(n)
    }
}