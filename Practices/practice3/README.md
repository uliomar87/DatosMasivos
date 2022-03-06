## CREATION OF LISTS IN SPARK

1. Create a list called "list" with the elements "red", "white", "black"  
   
    var Lista=List("red","white","black")

2. Add 5 more items to "list" "green" ,"yellow", "blue", "orange", "pearl"  

    Lista = "verde"::"amarillo"::"azul"::"naranja"::"perla"::Lista

3. Bring the items from "list" "green", "yellow", "blue"  

    Lista slice(0,3)

4. Create an array of numbers in the range 1-1000 in steps of 5 at a time  

    Array.range(1,1000,5)

5. What are the unique elements of the list List(1,3,3,4,6,7,3,7) use conversion to sets  

    var Lista1=List(1,3,3,4,6,7,3,7)

    val unico=List(1,3,3,4,6,7,3,7).toSet

6. Create a mutable map named names containing the following
     "Jose", 20, "Louis", 24, "Anna", 23, "Susana", "27"  
  
    val Nombres=collection.mutable.Map(("Jose",20),("Luis",20)("Ana",23)("Susana",27))  
  
   6 a.m. Print all the keys on the map  

    Nombres.keys
   
   6b. Add the following value to the map("Miguel", 23)  

    Nombres += ("Miguel" ->23)