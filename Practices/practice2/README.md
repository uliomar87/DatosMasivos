## Practice 2

1. Develop a scaling algorithm that calculates the radius of a circle
val diametro : Double = 20
val radio = diametro / 2
radio
<hr>

2. Develop an algorithm in scala that tells me if a number is prime
val num : Int = 10
var primo : Boolean = true
for(i <- Range(2, Num)) {
  if((Num % i) == 0) {
    primo = false
  }
}
if(primo){
  println("Es Primo")
} else {
  println("No es Primo")
}
<hr>

3. Given the variable var bird = "tweet", use string interpolation to print "Estoy ecribiendo un tweet"
var bird = "tweet"
printf("Estoy escribiendo un %s", bird)
<hr>

4. Given the variable var message = "Hola Luke yo soy tu padre!" uses slice to extract the "Luke" sequence
var mensaje = "Hola Luke yo soy tu padre!"
mensaje.slice(5,9)
<hr>

5. What is the difference between value (val) and a variable (var) in scala?
Var: The variable defined by var can be changed
Val: The variable defined by val cannot be changed
<hr>

6. Given the tuple (2,4,5,1,2,3,3.1416,23) it returns the number 3.1416
val tupla = (2,4,5,1,2,3,3.1416,23)
tupla._7
<hr>
