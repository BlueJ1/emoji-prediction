from typing import Optional
from config.database import Base, SessionLocal
from sqlalchemy.orm import Query as ORMQuery
from fastapi import HTTPException, Query

"""Dependencies
    Dependencies are used for Dependency Injection. They allow you to make
    reusable components that can be injected into your endpoints. This
    allows you to reuse the same code in multiple endpoints. For example,
    it allows you to have reusable query parameters that can be injected
    into multiple endpoints.
    This is what is being done with `OrderingParams` and `PaginationParams`.
    They contain a number of query parameters that are being used on other
    endpoints.


    More info: https://fastapi.tiangolo.com/tutorial/dependencies/
"""


def get_db():
    """
    This is a special dependency that is used to get the database session.

    It can be used directly on an endpoint like this:
    ```
    @router.get("/movies/{movie_id}")
    async def get_movie(
        movie_id: int,
        db: Session = Depends(get_db),
    ):
        # Do something with db
        ...
    ```

    But in this case I have used it in the controller dependency to
    avoid passing the db session to every function in the controller.
    Take a look in `/crud/controller.py` to see how it is used there.
    """

    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


class OrderingParams:
    """A Class that holds the Ordering Params"""

    def __init__(self, order_by: str, order_dir: str):
        self.order_by = order_by
        self.order_dir = order_dir

    def perform_ordering(self, query: ORMQuery, model: Base):
        """Performs the ordering on a query using the ordering parameters set.

        Args:
            query (ORMQuery): A query object that can be filtered.
            model (Base): Model that is being filtered

        Returns:
            ORMQuery : The filtered query
        """
        if (self.order_by is not None and self.order_by != ""):
            # Get the column from the model.
            # Equivalent to doing: Movie.title
            #   model = Movie
            #   title = self.order_by
            order_column = getattr(model, self.order_by)

            # Ordering is ascending by default.
            if self.order_dir == "desc":
                order_column = order_column.desc()

            # Above equivalent to: query.order_by(Movie.title.asc())
            query = query.order_by(order_column)
        return query


class OrderingParamsChecker:
    """Ordering Params Checker

    This class is used to validate the order_by and
    order_dir query parameters.
    It checks that:
        - If order_dir is passed, should either be "asc" or "desc"
        - If order_by is passed, should be one of the fields
            in the order_by_fields list.
    """

    def __init__(self, order_by_fields: list[str]):
        self.order_directions = ["asc", "desc"]
        self.order_by_fields = order_by_fields

    def raise_bad_request(detail: str):
        raise HTTPException(
            status_code=400,
            detail=detail,
        )

    def __call__(self,
                 order_dir: Optional[str] = Query(
                     None,
                     alias="order-dir",
                     description="The direction to order "
                                 "the indicated column", ),
                 order_by: Optional[str] = Query(
                     None,
                     alias="order-by",
                     description="The field to order the indicated column by.",
                 )):

        if order_by is None and order_dir is not None:
            raise self.raise_bad_request()

        if order_by is not None and order_dir is None:
            raise self.raise_bad_request()

        if (order_by is not None and order_by not in self.order_by_fields
                and order_by != ""):
            raise self.raise_bad_request()

        if order_dir is not None and order_dir not in self.order_directions:
            raise self.raise_bad_request()

        return OrderingParams(order_by=order_by, order_dir=order_dir)


class PaginationParams:
    """This class is used to handle the pagination
    parameters. It is injected into the endpoints
    as a dependency and then used to paginate the
    results.
    """

    def __init__(self, offset: int = Query(0), limit: int = Query()):
        # Offset is the number of records to skip. Defaults to 0
        self.offset = offset

        # Limit is the number of records to return. Is required.
        self.limit = limit

    def perform_pagination(self, query: ORMQuery):
        """A method used to paginate a query."""
        return query.offset(self.offset).limit(self.limit)
